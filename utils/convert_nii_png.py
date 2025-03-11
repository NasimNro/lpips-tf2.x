import numpy as np
import SimpleITK as sitk
import cv2
import argparse

def convert_nii_to_png(input_path: str, output_path: str, slice_idx: int = None) -> None:
    try:
        image = sitk.ReadImage(input_path)
        img_data = sitk.GetArrayFromImage(image)
        total_slices = img_data.shape[0]
        
        if slice_idx is None:
            slice_idx = total_slices // 2
            
        if slice_idx < 0 or slice_idx >= total_slices:
            raise ValueError(f"Slice index must be between 0 and {total_slices-1}")
            
        slice_data = img_data[slice_idx, :, :]
        slice_data = ((slice_data - slice_data.min()) / 
                     (slice_data.max() - slice_data.min()) * 255).astype(np.uint8)
        
        cv2.imwrite(output_path, slice_data)
        print(f"Successfully saved slice {slice_idx} to {output_path}")
        
    except Exception as e:
        print(f"Error during conversion: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert NII to PNG')
    parser.add_argument('input', help='Input NII file path')
    parser.add_argument('output', help='Output PNG file path')
    parser.add_argument('--slice', type=int, help='Slice index (default is middle slice)')
    
    args = parser.parse_args()
    convert_nii_to_png(args.input, args.output, args.slice) 