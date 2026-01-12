import numpy as np
import torch
import cv2
import logging
from mobile_sam import sam_model_registry, SamPredictor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SegmentationEngine:
    def __init__(self, checkpoint_path=None, model_type="vit_b", device=None, model_instance=None):
        """
        Initialize the SAM model.
        Args:
            checkpoint_path: Path to weights (if loading new).
            model_type: SAM architecture type.
            device: 'cuda' or 'cpu'.
            model_instance: Pre-loaded sam_model_registry instance (optional).
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        if model_instance is not None:
             self.sam = model_instance
        elif checkpoint_path:
             logger.info(f"Loading SAM model ({model_type}) on {self.device}...")
             self.sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
             self.sam.to(device=self.device)
        else:
             raise ValueError("Either checkpoint_path or model_instance must be provided.")

        self.predictor = SamPredictor(self.sam)
        self.is_image_set = False

    def set_image(self, image_rgb):
        """
        Process the image and compute embeddings.
        Args:
            image_rgb: NumPy array (H, W, 3) in RGB format.
        """
        logger.info("Computing image embeddings...")
        self.image_rgb = image_rgb # Store for safety checks
        self.predictor.set_image(image_rgb)
        self.is_image_set = True
        logger.info("Embeddings computed.")

    def generate_mask(self, point_coords, point_labels=None, level=None, cleanup=True):
        """
        Generate a mask for a given point.
        Args:
            point_coords: List of [x, y] or NumPy array.
            point_labels: List of labels (1 for foreground, 0 for background).
            level: int (0, 1, 2) or None. 
                   0=Fine Details, 1=Sub-segment, 2=Whole Object. 
                   If None, auto-selects highest score.
            cleanup: bool. If True, removes disconnected components to prevent leaks.
        """
        if not self.is_image_set:
            raise RuntimeError("Image not set. Call set_image() first.")

        if point_labels is None:
            point_labels = [1] * len(point_coords)

        point_coords = np.array(point_coords)
        point_labels = np.array(point_labels)

        with torch.inference_mode():
            masks, scores, logits = self.predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=True # Generate multiple masks and choose best
            )

        if level is not None and 0 <= level <= 2:
            # User forced a specific level
            best_mask = masks[level]
        else:
            # Heuristic: Favor 'Sub-segment' (Index 1) for architectural surfaces.
            # Index 0 is often too small (part of a wall), Index 2 is often too large (whole room).
            # Index 1 is the 'sweet spot' for walls/ceilings/floors.
            # We pick index 1 unless index 0 or 2 has a significantly higher confidence boost.
            if scores[1] > 0.70: # Lowered from 0.85 to be more selective
                best_mask = masks[1]
            else:
                best_idx = np.argmax(scores)
                best_mask = masks[best_idx]
        
        if cleanup:
            # Post-processing: Filter disconnected components
            # We only want the component that contains the clicked point.
            
            # Ensure mask is uint8 for OpenCV
            mask_uint8 = (best_mask * 255).astype(np.uint8)
            # --- SMART COLOR SAFETY CHECK ---
            # Re-enabled with Chromaticity Logic to fix Leaking AND Shadows.
            if hasattr(self, 'image_rgb'):
                cx, cy = int(point_coords[0][0]), int(point_coords[0][1])
                h, w = self.image_rgb.shape[:2]
                cx, cy = max(0, min(cx, w-1)), max(0, min(cy, h-1))
                
                # Sample Seed
                seed_color = self.image_rgb[cy, cx].astype(np.float32)
                
                # Check if seed is Grayscale (Saturation check)
                seed_mean = np.mean(seed_color)
                seed_sat = np.max(seed_color) - np.min(seed_color)
                is_grayscale_seed = seed_sat < 20 # Low saturation
                
                # 1. Chromaticity (Color only, invariant to brightness/shadows)
                # OPTIMIZATION: Use uint16 for distance check to avoid heavy float32 conversion
                img_u16 = self.image_rgb.astype(np.uint16)
                img_sum = np.sum(img_u16, axis=2, keepdims=True)
                img_sum[img_sum == 0] = 1 # Prevent div by zero
                
                # We can do this on integer space or float32? 
                # Float32 is better for precision, but let's do it faster.
                img_chroma = (img_u16[:, :, :2] << 8) // img_sum # Fixed point shift
                seed_chroma = (seed_color[:2].astype(np.uint16) << 8) // np.sum(seed_color + 0.1)
                
                # Color Distance
                chroma_dist = np.sum(np.abs(img_chroma - seed_chroma), axis=2)
                
                # 2. Intensity (Brightness)
                intensity_dist = np.abs(np.mean(img_u16, axis=2) - np.mean(seed_color))
                
                # 3. Hybrid Thresholding (ADAPTIVE BASED ON MODE)
                if level == 2: # "Whole Object" - UNLOCKED
                    valid_mask = np.ones((h, w), dtype=np.uint8)
                elif level == 0: # "Fine Detail" - SURGICAL BUT SHADOW-RESISTANT
                    # Loosened intensity to handle wall shadows while keeping chroma strict for leakes
                    if is_grayscale_seed:
                        valid_mask = (intensity_dist < 70).astype(np.uint8)
                    else:
                        valid_mask = ((chroma_dist < 28) & (intensity_dist < 110)).astype(np.uint8)
                else: # "Optimized" - BALANCED
                    if is_grayscale_seed:
                        valid_mask = (intensity_dist < 100).astype(np.uint8)
                    else:
                        valid_mask = ((chroma_dist < 40) & (intensity_dist < 160)).astype(np.uint8)

                # --- ULTRA-PRECISION EDGE GUARD (MODE-SENSITIVE) ---
                if level == 2:
                    edge_barrier = np.ones((h, w), dtype=np.uint8)
                else:
                    # HEAVY blur to ignore wall textures/wood grain but keep architectural lines
                    edge_gray = cv2.GaussianBlur(cv2.cvtColor(self.image_rgb, cv2.COLOR_RGB2GRAY), (9,9), 0)
                    
                    grad_x = cv2.Sobel(edge_gray, cv2.CV_16S, 1, 0, ksize=3)
                    grad_y = cv2.Sobel(edge_gray, cv2.CV_16S, 0, 1, ksize=3)
                    abs_grad_x = cv2.convertScaleAbs(grad_x)
                    abs_grad_y = cv2.convertScaleAbs(grad_y)
                    sobel_edges = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
                    
                    laplacian = cv2.Laplacian(edge_gray, cv2.CV_16S, ksize=3)
                    abs_laplacian = cv2.convertScaleAbs(laplacian)
                    edges = cv2.addWeighted(sobel_edges, 0.7, abs_laplacian, 0.3, 0)
                    
                    # Fine Detail uses 45, Optimized uses 60 (to ignore even more shadow edges)
                    e_thresh = 45 if level == 0 else 60
                    _, edge_barrier = cv2.threshold(edges, e_thresh, 255, cv2.THRESH_BINARY_INV)
                    edge_barrier = (edge_barrier / 255).astype(np.uint8)
                    
                    # Robust barrier thickening
                    edge_barrier = cv2.erode(edge_barrier, np.ones((3, 3), np.uint8), iterations=2)
                    
                    # SAFETY Zone around click (ensure paint starts smoothly)
                    cv2.circle(edge_barrier, (cx, cy), 15, 1, -1)

                # Intersect SAM mask with Adaptive Boundaries
                mask_refined = (mask_uint8 & valid_mask & edge_barrier)
                
                # --- LEAK PROTECTOR: Morphological cleanup to break tiny bridges ---
                kernel = np.ones((3, 3), np.uint8)
                mask_refined = cv2.morphologyEx(mask_refined, cv2.MORPH_OPEN, kernel)
                
                if np.sum(mask_refined) > 50: # At least some pixels survived
                    mask_uint8 = mask_refined
            
            # Check if the click point is actually inside the mask
            if len(point_coords) > 0:
                cx, cy = int(point_coords[0][0]), int(point_coords[0][1])
                
                # Find connected components (Connectivity=4 is stricter against leaks)
                num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(mask_uint8, connectivity=4)
                
                if num_labels > 1:
                    # labels_im has values 0 (bg), 1, 2, ...
                    # Get label at click position
                    # Make sure coordinates are within image bounds
                    h, w = mask_uint8.shape
                    cx = max(0, min(cx, w - 1))
                    cy = max(0, min(cy, h - 1))
                    
                    target_label = labels_im[cy, cx]
                    
                    if target_label != 0:
                        # Create a new mask keeping only the target component
                        best_mask = (labels_im == target_label)
                    else:
                        # Fallback: if click was somehow outside (e.g. edge case), keep largest component ignoring background
                        # stats[0] is background.
                        # Find max area among others
                        max_area = 0
                        max_label = 1
                        for i in range(1, num_labels):
                            if stats[i, cv2.CC_STAT_AREA] > max_area:
                                max_area = stats[i, cv2.CC_STAT_AREA]
                                max_label = i
                        best_mask = (labels_im == max_label)
        
        return best_mask
