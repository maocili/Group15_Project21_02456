import os
import cv2
import numpy as np
import tifffile as tiff
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def batch_process_masks_with_exceptions(src_folder, dst_folder, strict_factor=0.7, exception_files=[]):
    """
    batch process Mask with threshold(calculated by Otsu algorithem) = strict_factor * threshold，but exceptions with threshold = threshold
    param:
        exception_files: files produced with threshold = threshold
    """
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)

    # get files
    file_patterns = [os.path.join(src_folder, "*.tif"), os.path.join(src_folder, "*.tiff")]
    files = []
    for pattern in file_patterns:
        files.extend(glob(pattern))
    files.sort() # ensure the order

    print(f"(Factor={strict_factor})")
    print(f"(Factor=1.0) -> {exception_files}\n")

    for filepath in tqdm(files, desc="Processing"):
        try:
            filename = os.path.basename(filepath)

            # Standardization
            raw_data = tiff.imread(filepath)
            if raw_data.ndim == 3: raw_data = raw_data[:, :, 0]

            img_float = raw_data.astype(np.float32)
            min_val, max_val = np.min(img_float), np.max(img_float)

            if max_val - min_val == 0:
                img_uint8 = np.zeros_like(img_float, dtype=np.uint8)
            else:
                img_uint8 = ((img_float - min_val) / (max_val - min_val) * 255).astype(np.uint8)

            # Set the factor
            current_factor = strict_factor

            # if file in the exceptions，set the factor to 1.0
            if filename in exception_files:
                current_factor = 1.0
                tqdm.write(f" -> Applying factor = 1.0 to {filename}")

            # Calculate the threshold and tansfer to binary mask
            otsu_thresh, _ = cv2.threshold(
                img_uint8, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
            )

            final_thresh = otsu_thresh * current_factor

            # background = 0, foreground = 1)
            _, final_mask = cv2.threshold(
                img_uint8, final_thresh, 1, cv2.THRESH_BINARY_INV
            )

            # save
            tiff.imwrite(os.path.join(dst_folder, filename), final_mask.astype(np.uint8))

        except Exception as e:
            print(f"Error {filename} : {e}")



def compare_segmentation_methods(src_folder, image_names, strict_factor=0.7):
    """
    Compare different segmentation methods
    param:
        src_folder: file of original mask
        image_names: name of images for comparison
        strict_factor: strict factor of Otsu algorithem
        fixed_threshold: fixed threshold
    """
    n_images = len(image_names)
    fig = plt.figure(figsize=(15, 5 * n_images))
    gs = GridSpec(n_images, 4, figure=fig, hspace=0.3, wspace=0.3)
    
    for idx, filename in enumerate(image_names):
        filepath = os.path.join(src_folder, filename)
        
        try:
            # Standardization
            raw_data = tiff.imread(filepath)
            if raw_data.ndim == 3: raw_data = raw_data[:, :, 0]
            
            img_float = raw_data.astype(np.float32)
            min_val, max_val = np.min(img_float), np.max(img_float)
            
            if max_val - min_val == 0:
                img_uint8 = np.zeros_like(img_float, dtype=np.uint8)
            else:
                img_uint8 = ((img_float - min_val) / (max_val - min_val) * 255).astype(np.uint8)
            
            # method 1: fixed threshold segmentation
            # fixed_mask = np.zeros_like(img_uint8, dtype=np.uint8)
            # fixed_mask[img_uint8 > 250] = 1  
            # fixed_mask[img_uint8 < 3] = 0    
            fixed_mask = ((img_uint8 >= 3) & (img_uint8 <= 250)).astype(np.uint8)

            fixed_mask = cv2.inRange(img_uint8, 3, 250)
            fixed_mask = (255-fixed_mask) // 255
            
            # method 2: Otsu algorithem
            otsu_thresh, _ = cv2.threshold(
                img_uint8, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
            )
            
            # method 3: Otsu algorithem with strict_factor
            final_thresh = otsu_thresh * strict_factor
            _, adaptive_mask = cv2.threshold(
                img_uint8, final_thresh, 1, cv2.THRESH_BINARY_INV
            )
            
            # plot
            # original mask
            ax1 = fig.add_subplot(gs[idx, 0])
            ax1.imshow(img_uint8, cmap='gray')
            ax1.set_title(f'{filename}\nOriginal Mask')
            ax1.axis('off')
            
            # fixed threshold
            ax2 = fig.add_subplot(gs[idx, 1])
            ax2.imshow(fixed_mask, cmap='gray')
            ax2.set_title(f'Fixed Threshold\n(threshold=3, 250)')
            ax2.axis('off')
            
            # Otsu algorithem
            _, otsu_mask = cv2.threshold(
                img_uint8, otsu_thresh, 1, cv2.THRESH_BINARY_INV
            )
            ax3 = fig.add_subplot(gs[idx, 2])
            ax3.imshow(otsu_mask, cmap='gray')
            ax3.set_title(f'Otsu Method\n(threshold={otsu_thresh:.1f})')
            ax3.axis('off')
            
            # Otsu algorithem with strict_factor
            ax4 = fig.add_subplot(gs[idx, 3])
            ax4.imshow(adaptive_mask, cmap='gray')
            ax4.set_title(f'Adaptive Otsu\n(threshold={final_thresh:.1f}, factor={strict_factor})')
            ax4.axis('off')
            
            
        except Exception as e:
            print(f"Error {filename} : {e}")
    
    plt.suptitle('Segmentation Methods Comparison', fontsize=16, y=0.995)
    plt.tight_layout()
    plt.savefig('segmentation_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig


if __name__ == "__main__":

    input_dir = "data/Original Masks"
    output_dir = "data/Masks"

    special_files_list = ["image_v2_mask_00.tif"]
    '''
    batch_process_masks_with_exceptions(
        input_dir,
        output_dir,
        strict_factor=0.7,
        exception_files=special_files_list
    )

    '''
    images_to_compare = ["image_v2_mask_00.tif", "image_v2_mask_05.tif", "image_v2_mask_09.tif"]
    
    compare_segmentation_methods(
        input_dir, 
        images_to_compare, 
        strict_factor=0.7
    )
