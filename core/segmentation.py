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
        self.predictor.set_image(image_rgb)
        self.is_image_set = True
        logger.info("Embeddings computed.")
        self.image_rgb = image_rgb # Store for cleanup logic

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

        # Select best mask
        if level is not None and 0 <= level < 3:
            # User forced a specific level
            best_mask = masks[level]
        else:
            # Heuristic: Favor 'Sub-segment' (Index 1) for architectural surfaces.
            if scores[1] > 0.70: 
                best_mask = masks[1]
            else:
                best_idx = np.argmax(scores)
                best_mask = masks[best_idx]
        
        if cleanup:
            # Post-processing: Filter disconnected components
            h, w = best_mask.shape
            mask_uint8 = (best_mask * 255).astype(np.uint8)
            
            # --- SMART COLOR SAFETY CHECK ---
            # If we have a positive click, ensure we don't bleed into vastly different colors.
            # This is critical for White Wall -> White Cabinet separation.
            if len(point_coords) > 0 and len(point_labels) > 0:
                # Find the positive click (label 1)
                pos_indices = np.where(point_labels == 1)[0]
                if len(pos_indices) > 0:
                    idx = pos_indices[-1] # Use most recent click
                    cx, cy = int(point_coords[idx][0]), int(point_coords[idx][1])
                    
                    # Sample seed color (3x3 average for stability)
                    y1, y2 = max(0, cy-1), min(h, cy+2)
                    x1, x2 = max(0, cx-1), min(w, cx+2)
                    seed_patch = self.image_rgb[y1:y2, x1:x2]
                    seed_color = np.mean(seed_patch, axis=(0, 1))
                    
                    # Check for Grayscale Seed (White/Grey walls)
                    # If R~G~B, we tighten intensity limits and ignore chroma
                    std_dev = np.std(seed_color)
                    is_grayscale_seed = std_dev < 10.0 # Strict check for neutral colors
                    
                    # 1. Chroma (Color) Distance (Fast integer math)
                    img_u16 = self.image_rgb.astype(np.uint16)
                    img_sum = np.sum(img_u16, axis=2) + 1 # Avoid div/0
                    
                    # Normalize chromaticity: r = R/Sum, g = G/Sum
                    img_chroma = (img_u16[:, :, :2] << 8) // img_sum.reshape(h, w, 1) # Fixed point shift
                    seed_sum = np.sum(seed_color) + 0.1
                    seed_chroma = (seed_color[:2].astype(np.uint16) << 8) // int(seed_sum)
                    
                    # Color Distance
                    chroma_dist = np.sum(np.abs(img_chroma - seed_chroma), axis=2)
                    
                    # 2. Intensity (Brightness)
                    intensity_dist = np.abs(np.mean(img_u16, axis=2) - np.mean(seed_color))
                    
                    # 3. Hybrid Thresholding (ADAPTIVE BASED ON MODE)
                    if level == 2: # "Whole Object" 
                        valid_mask = np.ones((h, w), dtype=np.uint8)
                    elif level == 0: # "Fine Detail" (NOW DEFAULT) - SURGICAL
                        # Relax intensity heavily (90->120, 130->180) to allow painting over bright lights
                        # We rely on strict chroma (28) and edge barriers to stop leaks.
                        if is_grayscale_seed:
                            valid_mask = (intensity_dist < 120).astype(np.uint8)
                        else:
                            valid_mask = ((chroma_dist < 28) & (intensity_dist < 180)).astype(np.uint8)
                    else: # "Optimized" - BALANCED
                        # Consistent relaxation for fallback mode
                        if is_grayscale_seed:
                            valid_mask = (intensity_dist < 130).astype(np.uint8)
                        else:
                            valid_mask = ((chroma_dist < 30) & (intensity_dist < 190)).astype(np.uint8)

                    # --- ULTRA-PRECISION EDGE GUARD (MODE-SENSITIVE) ---
                    if level == 2:
                        # RUG & FLOOR PROTECTOR
                        kernel_ext = np.ones((7, 7), np.uint8)
                        mask_refined = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel_ext)
                        mask_refined = (mask_refined & valid_mask)
                        edge_barrier = np.ones((h, w), dtype=np.uint8)
                    else:
                        # Sharpened blur (9x9 -> 5x5) to catch Cabinet & Cornice lines in Auto-Detect
                        if level == 0:
                             k_size = (9, 9)
                        else:
                             k_size = (9, 9)
                        
                        edge_gray = cv2.GaussianBlur(cv2.cvtColor(self.image_rgb, cv2.COLOR_RGB2GRAY), k_size, 0)
                        
                        grad_x = cv2.Sobel(edge_gray, cv2.CV_16S, 1, 0, ksize=3)
                        grad_y = cv2.Sobel(edge_gray, cv2.CV_16S, 0, 1, ksize=3)
                        abs_grad_x = cv2.convertScaleAbs(grad_x)
                        abs_grad_y = cv2.convertScaleAbs(grad_y)
                        sobel_edges = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
                        
                        laplacian = cv2.Laplacian(edge_gray, cv2.CV_16S, ksize=3)
                        abs_laplacian = cv2.convertScaleAbs(laplacian)
                        edges = cv2.addWeighted(sobel_edges, 0.7, abs_laplacian, 0.3, 0)
                        
                        # Fine Detail matches Optimized at 45
                        e_thresh = 45 
                        _, edge_barrier = cv2.threshold(edges, e_thresh, 255, cv2.THRESH_BINARY_INV)
                        edge_barrier = (edge_barrier / 255).astype(np.uint8)
                        
                        # Robust barrier thickening
                        edge_barrier = cv2.erode(edge_barrier, np.ones((3, 3), np.uint8), iterations=2)
                        
                        # SAFETY Zone around click (ensure paint starts smoothly)
                        cv2.circle(edge_barrier, (cx, cy), 15, 1, -1)

                    # Intersect SAM mask with Adaptive Boundaries
                    if level != 2:
                        mask_refined = (mask_uint8 & valid_mask & edge_barrier)
                    else:
                        mask_refined = mask_uint8 # For level 2, valid_mask is all ones, edge_barrier is all ones.
                    
                    # --- LEAK PROTECTOR: Morphological cleanup to break tiny bridges ---
                    kernel = np.ones((3, 3), np.uint8)
                    mask_refined = cv2.morphologyEx(mask_refined, cv2.MORPH_OPEN, kernel)
                    
                    if np.sum(mask_refined) > 50: # At least some pixels survived
                        mask_uint8 = mask_refined
            
            # Check if the click point is actually inside the mask (it should be, but just in case)
            # We take the first point (positive click)
            if len(point_coords) > 0:
                cx, cy = int(point_coords[0][0]), int(point_coords[0][1])
                
                # Find connected components
                num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)
                
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
