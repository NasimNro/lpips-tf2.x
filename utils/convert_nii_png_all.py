import numpy as np
import SimpleITK as sitk
import cv2
import argparse
import os

def convert_nii_to_png_all_slices(input_path: str, output_dir: str) -> None:
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Read the NIfTI file
        image = sitk.ReadImage(input_path)
        img_data = sitk.GetArrayFromImage(image)
        total_slices = img_data.shape[0]
        
        print(f"Converting {total_slices} slices...")
        
        # Process each slice
        for slice_idx in range(total_slices):
            # Get the slice data
            slice_data = img_data[slice_idx, :, :]
            
            # Normalize to 0-255 range
            slice_data = ((slice_data - slice_data.min()) / 
                         (slice_data.max() - slice_data.min()) * 255).astype(np.uint8)
            
            # Create output filename with padding (e.g., slice_000.png)
            output_path = os.path.join(output_dir, f'slice_{slice_idx:03d}.png')
            
            # Save the slice
            cv2.imwrite(output_path, slice_data)
            print(f"Saved slice {slice_idx}/{total_slices-1} to {output_path}")
        
        print(f"\nSuccessfully converted all {total_slices} slices to PNG files in {output_dir}")
        
    except Exception as e:
        print(f"Error during conversion: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert all slices of a NIfTI file to PNG images')
    parser.add_argument('input', help='Input NIfTI file path')
    parser.add_argument('output_dir', help='Output directory for PNG files')
    
    args = parser.parse_args()
    convert_nii_to_png_all_slices(args.input, args.output_dir) 