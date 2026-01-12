import cv2
import numpy as np
from PIL import Image

class ColorTransferEngine:
    @staticmethod
    def hex_to_rgb(hex_color):
        """Convert HEX string to RGB tuple."""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

    @staticmethod
    def apply_color(image_rgb, mask, target_color_hex, intensity=1.0):
        """
        Apply color with improved texture preservation using Luminosity blending simulation.
        """
        # Ensure input is standard format
        image_rgb = image_rgb.astype(np.uint8)
        
        # 1. Create Smooth Mask (Feathering)
        mask_float = mask.astype(np.float32)
        # Gentle blur for anti-aliasing edges
        mask_soft = cv2.GaussianBlur(mask_float, (5, 5), 0)
        mask_3ch = np.stack([mask_soft] * 3, axis=-1)
        
        # 2. Prepare Target Color
        target_rgb = ColorTransferEngine.hex_to_rgb(target_color_hex)
        target_bgr = np.array([target_rgb[2], target_rgb[1], target_rgb[0]], dtype=np.uint8)
        
        # Create a solid color image of the same size
        colored_layer = np.full_like(image_rgb, target_rgb, dtype=np.uint8)
        
        # 3. Apply Color with Luminosity blending (LAB swap + L-Shift)
        img_float = image_rgb.astype(np.float32) / 255.0
        img_lab = cv2.cvtColor(img_float, cv2.COLOR_RGB2Lab)
        L, A, B = cv2.split(img_lab)
        
        # Target color in LAB
        target_pixel = np.array([[[target_rgb[0], target_rgb[1], target_rgb[2]]]], dtype=np.uint8)
        target_lab = cv2.cvtColor(target_pixel.astype(np.float32) / 255.0, cv2.COLOR_RGB2Lab)
        target_l, target_a, target_b = target_lab[0, 0]
        
        # Preserve texture but shift brightness toward target
        # Calculate mean L of the original masked area
        masked_L = L[mask > 0]
        if masked_L.size > 0:
            mean_orig_l = np.mean(masked_L)
            l_shift = (target_l - mean_orig_l) * 0.7 
            L = L + l_shift
        
        new_A = np.full_like(A, target_a)
        new_B = np.full_like(B, target_b)
        
        new_lab = cv2.merge([L, new_A, new_B])
        recolored_rgb = cv2.cvtColor(new_lab, cv2.COLOR_Lab2RGB)
        
        # 4. Blend based on mask
        result_float = (recolored_rgb * mask_3ch) + (img_float * (1.0 - mask_3ch))
        
        result_uint8 = np.clip(result_float * 255.0, 0, 255).astype(np.uint8)
        
        return result_uint8

    @staticmethod
    def composite_multiple_layers(image_rgb, masks_data):
        """
        Apply multiple colored masks and textures.
        Solid colors are processed together in LAB space for speed.
        Textures are applied individually.
        """
        if not masks_data:
            return image_rgb.copy()

        # 1. Split into Solid vs Texture
        solids = [d for d in masks_data if d.get('color') and d.get('texture') is None]
        textures = [d for d in masks_data if d.get('texture') is not None]

        # 2. Process Solids in a single LAB pass
        current_result = image_rgb.astype(np.float32) / 255.0
        
        if solids:
            img_lab = cv2.cvtColor(current_result, cv2.COLOR_RGB2Lab)
            L, A, B = cv2.split(img_lab)
            curr_A = A.copy()
            curr_B = B.copy()

            for data in solids:
                # PERFORMANCE: Use cached soft mask if available, but VALIDATE SHAPE
                # (Prevents crash if high-res export tries to use preview-sized cache)
                cached_soft = data.get('mask_soft')
                if cached_soft is not None and cached_soft.shape[:2] == L.shape:
                    mask_soft = cached_soft
                else:
                    mask_float = data['mask'].astype(np.float32)
                    mask_soft = cv2.GaussianBlur(mask_float, (5, 5), 0)
                    data['mask_soft'] = mask_soft 

                # Global Layer Opacity
                layer_opacity = data.get('opacity', 1.0)
                mask_soft = mask_soft * layer_opacity

                # Prepare Target Color A/B
                rgb = ColorTransferEngine.hex_to_rgb(data['color'])
                target_pixel = np.array([[[rgb[0], rgb[1], rgb[2]]]], dtype=np.uint8)
                target_lab = cv2.cvtColor(target_pixel.astype(np.float32) / 255.0, cv2.COLOR_RGB2Lab)
                target_l = target_lab[0, 0, 0]
                target_a = target_lab[0, 0, 1]
                target_b = target_lab[0, 0, 2]

                # 1. Blend A/B channels (The color tint)
                curr_A = (target_a * mask_soft) + (curr_A * (1.0 - mask_soft))
                curr_B = (target_b * mask_soft) + (curr_B * (1.0 - mask_soft))

                # 2. MATCH LUMINANCE (The "Paint Coverage" effect)
                # We want the surface to move toward the target color's brightness 
                # while keeping its texture (the variations in L).
                masked_L = L[data['mask'] > 0]
                if masked_L.size > 0:
                    mean_orig_l = np.mean(masked_L)
                    # Calculate how much to shift the brightness
                    # We use a 70% factor to avoid completely crushing original lighting/shadows
                    l_shift = (target_l - mean_orig_l) * 0.7 
                    L = L + (l_shift * mask_soft)

                # Lighting Adjustment (Brightness/Contrast/Saturation/Hue)
                brightness = data.get('brightness', 0.0)
                contrast = data.get('contrast', 1.0)
                saturation = data.get('saturation', 1.0) # 0.0 to 2.0
                hue_shift = data.get('hue', 0.0)         # -20 to 20
                
                if brightness != 0.0 or contrast != 1.0:
                    # Adjust L channel within mask
                    L_adjusted = L * contrast + brightness
                    L = (L_adjusted * mask_soft) + (L * (1.0 - mask_soft))
                
                if saturation != 1.0 or hue_shift != 0.0:
                    curr_A = (curr_A * saturation + hue_shift) * mask_soft + curr_A * (1.0 - mask_soft)
                    curr_B = (curr_B * saturation) * mask_soft + curr_B * (1.0 - mask_soft)

                # Finish Mode Adjustment
                finish_mode = data.get('finish', 'Standard')
                if finish_mode == "Matte":
                    # Reduce highlights only within the mask
                    L_matte = np.where(L > 80, 80 + (L - 80) * 0.3, L)
                    L = (L_matte * mask_soft) + (L * (1.0 - mask_soft))
                elif finish_mode == "Glossy":
                    # Boost highlights and midtones only within the mask
                    L_glossy = np.where(L > 60, L * 1.1, L)
                    L = (L_glossy * mask_soft) + (L * (1.0 - mask_soft))
            
            # Re-merge ONCE after all solids are processed
            new_lab = cv2.merge([L, curr_A, curr_B])
            current_result = cv2.cvtColor(new_lab, cv2.COLOR_Lab2RGB)

        # 3. Process Textures individually on top
        # (Textures replace the color structure so they must be applied sequentially)
        result_uint8 = np.clip(current_result * 255.0, 0, 255).astype(np.uint8)
        
        for data in textures:
            # PERFORMANCE: Handle soft mask caching with shape validation
            cached_soft = data.get('mask_soft')
            if cached_soft is None or cached_soft.shape[:2] != result_uint8.shape[:2]:
                mask_float = data['mask'].astype(np.float32)
                data['mask_soft'] = cv2.GaussianBlur(mask_float, (3, 3), 0)
                cached_soft = data['mask_soft']

            result_uint8 = ColorTransferEngine.apply_texture(
                result_uint8, 
                data['mask'], 
                data['texture'], 
                opacity=data.get('opacity', 0.8),
                brightness=data.get('brightness', 0.0),
                contrast=data.get('contrast', 1.0),
                saturation=data.get('saturation', 1.0),
                hue_shift=data.get('hue', 0.0),
                rotation=data.get('tex_rot', 0),
                scale_adj=data.get('tex_scale', 1.0),
                mask_soft=cached_soft
            )
            
        return result_uint8

    @staticmethod
    def apply_texture(image_rgb, mask, texture_rgb, opacity=0.8, brightness=0.0, contrast=1.0, saturation=1.0, hue_shift=0.0, rotation=0, scale_adj=1.0, mask_soft=None):
        """
        Apply a texture with perspective and blending.
        """
        image_rgb = image_rgb.astype(np.uint8)
        
        # 1. PERFORMANCE: Use cached soft mask if available
        if mask_soft is None:
            mask_float = mask.astype(np.float32)
            mask_soft = cv2.GaussianBlur(mask_float, (3, 3), 0)

        mask_3ch = np.stack([mask_soft] * 3, axis=-1)
        
        # 2. Tile and Transform Texture
        h, w, c = image_rgb.shape
        
        # Rotate/Scale the texture source
        if rotation != 0 or scale_adj != 1.0:
            th, tw = texture_rgb.shape[:2]
            center = (tw // 2, th // 2)
            M = cv2.getRotationMatrix2D(center, rotation, scale_adj)
            texture_rgb = cv2.warpAffine(texture_rgb, M, (tw, th))

        th, tw, tc = texture_rgb.shape
        if max(th, tw) > 512:
            s = 512 / max(th, tw)
            texture_rgb = cv2.resize(texture_rgb, (0, 0), fx=s, fy=s)
            th, tw, tc = texture_rgb.shape
            
        tiled_texture = np.zeros_like(image_rgb)
        for i in range(0, h, th):
            for j in range(0, w, tw):
                curr_h = min(th, h - i)
                curr_w = min(tw, w - j)
                tiled_texture[i:i+curr_h, j:j+curr_w] = texture_rgb[:curr_h, :curr_w]
        
        # Apply Lighting Adjustments
        if brightness != 0.0 or contrast != 1.0 or saturation != 1.0 or hue_shift != 0.0:
            tex_float = tiled_texture.astype(np.float32) / 255.0
            tex_lab = cv2.cvtColor(tex_float, cv2.COLOR_RGB2Lab)
            tL, tA, tB = cv2.split(tex_lab)
            tL = tL * contrast + brightness
            tA = tA * saturation + hue_shift
            tB = tB * saturation
            tex_lab = cv2.merge([tL, tA, tB])
            tiled_texture = (cv2.cvtColor(tex_lab, cv2.COLOR_Lab2RGB) * 255).astype(np.uint8)

        # 3. Blend texture with original luminosity (Texture over luminance)
        # Using a soft overlay style blending
        blended = tex_float * (gray_3ch * 1.2) 
        blended = np.clip(blended, 0, 1.0)
        
        # 4. Composite
        final_mask = mask_3ch * opacity
        output = (blended * final_mask) + (img_float * (1.0 - final_mask))
        
        return np.clip(output * 255.0, 0, 255).astype(np.uint8)
