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

        # Force correct shapes (N, 2) and (N,)
        if point_coords.ndim == 1:
            point_coords = point_coords[None, :]
        if point_labels.ndim > 1:
            point_labels = point_labels.flatten()
            
        logger.info(f"SAM Predict Inputs: Coords={point_coords.shape}, Labels={point_labels.shape}")

        try:
            with torch.no_grad():
                masks, scores, logits = self.predictor.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    multimask_output=True 
                )
        except Exception as e:
            logger.error(f"SAM Prediction Failed: {e}")
            return None

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
            
            # 1. Hole Filling (Fix for "Spotty" Walls)
            kernel = np.ones((10,10), np.uint8)
            mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)

            # 2. Level-Specific Logic
            # "Fine Detail" -> Strict spatial and color control
            if level == 0 and hasattr(self, 'image_rgb'):
                 # A. Spatial Constraint: Flood Fill with Edge Barriers
                 cx, cy = int(point_coords[0][0]), int(point_coords[0][1])
                 h, w = self.image_rgb.shape[:2]
                 cx, cy = max(0, min(cx, w-1)), max(0, min(cy, h-1))
                 
                 gray = cv2.cvtColor(self.image_rgb, cv2.COLOR_RGB2GRAY)
                 edges = cv2.Canny(gray, 50, 150)
                 edges = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=1)
                 
                 fill_mask = np.zeros((h + 2, w + 2), np.uint8)
                 fill_mask[1:-1, 1:-1] = edges 
                 
                 # Only flood fill if we are not clicking ON an edge
                 if fill_mask[cy+1, cx+1] == 0:
                     # 15 tolerance
                     cv2.floodFill(
                         self.image_rgb.copy(), 
                         fill_mask, 
                         (cx, cy), 
                         255, 
                         (15, 15, 15), 
                         (15, 15, 15), 
                         flags=(4 | cv2.FLOODFILL_FIXED_RANGE | cv2.FLOODFILL_MASK_ONLY)
                     )
                     spatial_mask = (fill_mask[1:-1, 1:-1] == 255).astype(np.uint8) * 255
                     # Intersect
                     if np.any(spatial_mask):
                         mask_uint8 = cv2.bitwise_and(mask_uint8, spatial_mask)

            # --- PRE-FILTER: COLOR SAFETY CHECK (Fix for "Leaking") ---
            # If we have the original image, we can verify that the mask doesn't 
            # bleed into areas with wildly different colors (e.g., wall vs ceiling).
            if hasattr(self, 'image_rgb'):
                cx, cy = int(point_coords[0][0]), int(point_coords[0][1])
                h, w = self.image_rgb.shape[:2]
                cx, cy = max(0, min(cx, w-1)), max(0, min(cy, h-1))
                
                # Sample the "True" color at click point
                seed_color = self.image_rgb[cy, cx]
                
                # Convert image + seed to LAB for perceptual distance
                # (Doing full image conversion is expensive, so we just do it on masked area later if needed)
                # For speed, we use a simple RGB distance threshold first
                
                # Extract mask area pixels
                masked_pixels = self.image_rgb[mask_uint8 > 0]
                
                if masked_pixels.size > 0:
                    # More robust: Prune pixels that are TOO different from seed
                    # This cuts off "leaks" into dark shadows or bright lights
                    # 1. Calculate distance of all masked pixels from seed
                    diff = np.abs(self.image_rgb.astype(np.int16) - seed_color.astype(np.int16))
                    dist_mask = np.mean(diff, axis=2) # Average difference across R,G,B
                    
                    # Threshold: Allow variation for shading (60), but cut off distinct colors (>60)
                    # If it's "Whole Object", we allow more variation (80)
                    # UPDATED: Set Level 2 to 250 (Disable Check), Default to 150 (More tolerant)
                    safety_thresh = 250 if level == 2 else 150
                    
                    valid_color_mask = (dist_mask < safety_thresh).astype(np.uint8)
                    
                    # Intersect: Only keep parts of the AI mask that match the color roughly
                    # BUT enforce connectivity to the click point so we don't keep random noise
                    mask_refined = (mask_uint8 & valid_color_mask)
                    
                    # If this killed the mask (e.g. click was on a highlight), fallback to original
                    if np.sum(mask_refined) > 100:
                        mask_uint8 = mask_refined
            # Only apply if we have the image
            if hasattr(self, 'image_rgb'):
                # Sample color around click
                cx, cy = int(point_coords[0][0]), int(point_coords[0][1])
                h, w = self.image_rgb.shape[:2]
                pad = 5
                y1, y2 = max(0, cy-pad), min(h, cy+pad)
                x1, x2 = max(0, cx-pad), min(w, cx+pad)
                
                sample_region = self.image_rgb[y1:y2, x1:x2]
                if sample_region.size > 0:
                     # Flatten to (N, 3) to strictly compute mean color
                     sample_pixels = sample_region.reshape(-1, 3)
                     
                     mean_rgb = np.mean(sample_pixels, axis=0)
                     std_rgb = np.std(sample_pixels, axis=0)
                     
                     # Adaptive Sigma
                     sigma = 2.5 # Default (Balanced)
                     if level == 2: sigma = 6.0 # Whole -> Loose
                     elif level == 0: sigma = 2.0 # Fine -> Strict
                     
                     thresh_val = np.maximum(sigma * std_rgb, 25.0) # Min thresh 25
                     
                     lower = np.maximum(0, mean_rgb - thresh_val)
                     upper = np.minimum(255, mean_rgb + thresh_val)
                     
                     color_mask = cv2.inRange(self.image_rgb, lower.astype(np.uint8), upper.astype(np.uint8))
                     
                     # Intersect with existing mask
                     mask_uint8 = cv2.bitwise_and(mask_uint8, color_mask)

            # 4. Connected Component Analysis (Keep largest connected to click)
            num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(mask_uint8, connectivity=4)
            if num_labels > 1:
                cx, cy = int(point_coords[0][0]), int(point_coords[0][1])
                h, w = mask_uint8.shape
                cx, cy = max(0, min(cx, w - 1)), max(0, min(cy, h - 1))
                
                target_label = labels_im[cy, cx]
                
                # If click missed slightly, search nearby
                if target_label == 0:
                    dist = 1
                    found = False
                    while dist < 15 and not found:
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
                    # Fallback to largest area
                    best_mask = (labels_im == (1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])))
            else:
                 # Just use what we have if only 1 component (or 0?)
                 best_mask = (mask_uint8 > 127)
            
            # Final output is bool
            return best_mask.astype(bool)

        return best_mask.astype(bool)
