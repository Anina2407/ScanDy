"""
Script to resize and store object masks, images, and feature maps while preserving proportions.
All files maintain alignment for layering.
"""

import numpy as np
import os
from pathlib import Path
from skimage.transform import resize
from skimage.io import imread, imsave
from tqdm import tqdm
import cv2


# Configuration
INPUT_DIRS = {
    'objectmasks': 'PictureExample/polished_segmentation/',  # .npy files
    'images': 'PictureExample/picture/',             # .png files
    'featuremaps': 'PictureExample/featuremaps/'    # .npy files
}

OUTPUT_DIRS = {
    'objectmasks': 'PictureExample/PictureExample_Resized/polished_segmentation/',
    'images': 'PictureExample/PictureExample_Resized/picture/',
    'featuremaps': 'PictureExample/PictureExample_Resized/featuremaps/'
}

# Target size
TARGET_HEIGHT = 270  # VID_SIZE_Y
TARGET_WIDTH = 480   # VID_SIZE_X

def resize_objectmask(mask, target_shape):
    """
    Resize object mask preserving discrete object IDs.
    
    :param mask: Object mask array (can be 2D or 3D)
    :param target_shape: (height, width) tuple
    :return: Resized mask
    """
    target_h, target_w = target_shape
    
    # Handle 2D mask (single frame)
    if mask.ndim == 2:
        return cv2.resize(
            mask,
            (target_w, target_h), 
            interpolation=cv2.INTER_NEAREST
        ).astype(mask.dtype)
    
    # Handle 3D mask (multiple frames)
    elif mask.ndim == 3:
        resized = np.zeros((mask.shape[0], target_h, target_w),  dtype=mask.dtype)
        resized_frame = []
        for i in range(mask.shape[0]):
            resized[i] = cv2.resize(
                mask[i],
                (target_w, target_h),
                interpolation=cv2.INTER_NEAREST
            ).astype(mask.dtype)
            resized_frame.append(resized[i].astype(mask.dtype))
        resized = np.stack(resized_frame, axis=0)
        return resized
    
    else:
        raise ValueError(f"Unexpected mask dimensions: {mask.ndim}")


def resize_image(img, target_shape):
    """
    Resize image maintaining aspect ratio and quality.
    
    :param img: Image array (H, W) or (H, W, C)
    :param target_shape: (height, width) tuple
    :return: Resized image
    """
    target_h, target_w = target_shape
    
    resized = cv2.resize(
        img,
        (target_w, target_h),  # Note: OpenCV uses (width, height)
        interpolation=cv2.INTER_AREA
    )
    
    # Preserve original dtype
    if img.dtype == np.uint8:
        return np.clip(resized, 0, 255).astype(np.uint8)
    else:
        return resized.astype(img.dtype)

def resize_featuremap(fmap, target_shape):
    """
    Resize feature map with smooth interpolation.
    
    :param fmap: Feature map array (can be 2D or 3D)
    :param target_shape: (height, width) tuple
    :return: Resized feature map
    """
    target_h, target_w = target_shape
    
    # Handle 2D feature map (single frame)
    if fmap.ndim == 2:
        try:
            # Ensure proper dtype and contiguous array
            fmap_input = np.ascontiguousarray(fmap, dtype=np.float32)
            
            # Try with explicit copy to ensure memory is clean
            fmap_copy = fmap_input.copy()
            
            resized = cv2.resize(
                fmap_copy,
                (target_w, target_h),
                interpolation=cv2.INTER_AREA
            )
            return resized.astype(np.float32)
        except cv2.error as e:
            print(f"OpenCV error, falling back to scipy. Error: {e}")
            # Fallback to scipy
            from scipy.ndimage import zoom
            scale_h = target_h / fmap.shape[0]
            scale_w = target_w / fmap.shape[1]
            return zoom(fmap, (scale_h, scale_w), order=1).astype(np.float32)
    
    # Handle 3D feature map (multiple frames)
    elif fmap.ndim == 3:
        resized = np.zeros((fmap.shape[0], target_h, target_w), dtype=np.float32)
        for i in range(fmap.shape[0]):
            try:
                # Ensure proper dtype and contiguous array for each frame
                frame_input = np.ascontiguousarray(fmap[i], dtype=np.float32)
                
                # Try with explicit copy
                frame_copy = frame_input.copy()
                
                resized[i] = cv2.resize(
                    frame_copy,
                    (target_w, target_h),
                    interpolation=cv2.INTER_AREA
                )
            except cv2.error as e:
                if i == 0:  # Only print once
                    print(f"OpenCV error on frame {i}, using fallback. Error: {e}")
                # Fallback to scipy
                from scipy.ndimage import zoom
                scale_h = target_h / fmap[i].shape[0]
                scale_w = target_w / fmap[i].shape[1]
                resized[i] = zoom(fmap[i], (scale_h, scale_w), order=1).astype(np.float32)
        return resized
    
    else:
        raise ValueError(f"Unexpected feature map dimensions: {fmap.ndim}")
    
def process_objectmasks(input_dir, output_dir, target_shape):
    """Process all object mask .npy files."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    npy_files = list(input_path.glob('*.npy'))
    print(f"\nProcessing {len(npy_files)} object mask files...")
    
    for npy_file in tqdm(npy_files):
        mask = np.load(npy_file)
        resized_mask = resize_objectmask(mask, target_shape)
        
        output_file = output_path / npy_file.name
        np.save(output_file, resized_mask)


def process_images(input_dir, output_dir, target_shape):
    """Process all image .png files."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    png_files = list(input_path.glob('*.png'))
    print(f"\nProcessing {len(png_files)} image files...")
    
    for png_file in tqdm(png_files):
        img = imread(png_file)
        resized_img = resize_image(img, target_shape)
        
        output_file = output_path / png_file.name
        imsave(output_file, resized_img, check_contrast=False)


def process_featuremaps(input_dir, output_dir, target_shape):
    """Process all feature map .npy files (including subdirectories)."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Find all .npy files recursively
    npy_files = list(input_path.rglob('*.npy'))
    print(f"\nProcessing {len(npy_files)} feature map files...")
    
    for npy_file in tqdm(npy_files):
        # Maintain subdirectory structure
        relative_path = npy_file.relative_to(input_path)
        output_file = output_path / relative_path
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        fmap = np.load(npy_file)
        resized_fmap = resize_featuremap(fmap, target_shape)
        
        np.save(output_file, resized_fmap)


def main():
    target_shape = (TARGET_HEIGHT, TARGET_WIDTH)
    
    print(f"Resizing all files to: {TARGET_WIDTH}x{TARGET_HEIGHT}")
    print("=" * 60)
    
    # Process object masks
    if os.path.exists(INPUT_DIRS['objectmasks']):
        process_objectmasks(
            INPUT_DIRS['objectmasks'],
            OUTPUT_DIRS['objectmasks'],
            target_shape
        )
    else:
        print(f"Warning: {INPUT_DIRS['objectmasks']} not found, skipping.")
    
    # # Process images
    if os.path.exists(INPUT_DIRS['images']):
        process_images(
            INPUT_DIRS['images'],
            OUTPUT_DIRS['images'],
            target_shape
        )
    else:
        print(f"Warning: {INPUT_DIRS['images']} not found, skipping.")
    
    # Process feature maps
    if os.path.exists(INPUT_DIRS['featuremaps']):
        process_featuremaps(
            INPUT_DIRS['featuremaps'],
            OUTPUT_DIRS['featuremaps'],
            target_shape
        )
    else:
        print(f"Warning: {INPUT_DIRS['featuremaps']} not found, skipping.")
    
    print("\n" + "=" * 60)
    print("Resizing complete!")
    print(f"Output directories:")
    for name, path in OUTPUT_DIRS.items():
        print(f"  - {name}: {path}")


if __name__ == "__main__":
    main()