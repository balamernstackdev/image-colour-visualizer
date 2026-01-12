import numpy as np
import torch
import cv2
import logging
from segment_anything import sam_model_registry, SamPredictor

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
        self.image_rgb = image_rgb # Store for color-aware cleanup
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
        """
        if not self.is_image_set:
            raise RuntimeError("Image not set. Call set_image() first.")

        if point_labels is None:
            point_labels = [1] * len(point_coords)

        point_coords = np.array(point_coords)
        point_labels = np.array(point_labels)

        with torch.no_grad():
            masks, scores, logits = self.predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=True 
            )

        if level is not None and 0 <= level <= 2:
            best_mask = masks[level]
        else:
            # Heuristic: Prefer Index 1 for general surfaces
            if scores[1] > 0.8:
                 best_mask = masks[1]
            else:
                 best_idx = np.argmax(scores)
                 best_mask = masks[best_idx]
        
        if cleanup:
            mask_uint8 = (best_mask * 255).astype(np.uint8)
            
            # --- EDGELINE-AWARE FLOOD SELECTION ---
            if level == 0 and hasattr(self, 'image_rgb'):
                cx, cy = int(point_coords[0][0]), int(point_coords[0][1])
                h, w = self.image_rgb.shape[:2]
                cx, cy = max(0, min(cx, w-1)), max(0, min(cy, h-1))
                
                # 1. Create an Edge Barrier
                # This helps the flood fill stop at physical cracks/lines even if colors match
                gray = cv2.cvtColor(self.image_rgb, cv2.COLOR_RGB2GRAY)
                edges = cv2.Canny(gray, 50, 150)
                # Dilate edges slightly to make them a solid wall
                edges = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=1)
                
                # 2. Flood Fill with Edge Barrier
                # We use a 2px larger mask as per OpenCV spec
                fill_mask = np.zeros((h + 2, w + 2), np.uint8)
                # Initialize mask with edges
                fill_mask[1:-1, 1:-1] = edges 
                
                # Use a tighter tolerance for Fine Detail (15-18)
                tol = 15
                lo_diff = (tol, tol, tol)
                hi_diff = (tol, tol, tol)
                
                # Local copy of image for processing
                proc_img = self.image_rgb.copy()
                
                # Only perform fill if seed isn't on an edge
                if fill_mask[cy+1, cx+1] == 0:
                    cv2.floodFill(
                        proc_img, 
                        fill_mask, 
                        (cx, cy), 
                        255, # Value to fill in mask
                        lo_diff, 
                        hi_diff, 
                        flags=(4 | cv2.FLOODFILL_FIXED_RANGE | cv2.FLOODFILL_MASK_ONLY)
                    )
                
                # Extract the continuity mask (exclude the pre-existing edge barriers)
                # We just want pixels that were newly filled (value 255)
                color_continuity_mask = (fill_mask[1:-1, 1:-1] == 255).astype(np.uint8)
                
                # 3. Trim AI mask
                # If flood fill produced a valid result, use it. 
                # Otherwise (e.g. click was on an edge), fall back to tighter AI mask.
                if np.any(color_continuity_mask):
                    mask_uint8 = (mask_uint8 & (color_continuity_mask * 255).astype(np.uint8))

            # --- NO EROSION (Preserve Connection) ---
            # Instead of eroding (which breaks thin connections), we perform a 
            # stricter connected component analysis using 4-connectivity.
            
            num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(mask_uint8, connectivity=4)
            
            if num_labels > 1:
                cx, cy = int(point_coords[0][0]), int(point_coords[0][1])
                h, w = mask_uint8.shape
                cx, cy = max(0, min(cx, w - 1)), max(0, min(cy, h - 1))
                
                target_label = labels_im[cy, cx]
                
                # Search for nearest component if click point was missed
                if target_label == 0:
                    dist = 1
                    found = False
                    while dist < 12 and not found:
                        for dy in range(-dist, dist+1):
                            for dx in range(-dist, dist+1):
                                px, py = cx + dx, cy + dy
                                if 0 <= px < w and 0 <= py < h and labels_im[py, px] != 0:
                                    target_label = labels_im[py, px]; found = True; break
                            if found: break
                        dist += 1
                
                if target_label != 0:
                    best_mask = (labels_im == target_label)
                else:
                    best_mask = (labels_im == (1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])))
            
            # Post-processing: Soften edges slightly
            best_mask = best_mask.astype(np.uint8)
        
        return best_mask.astype(bool)
