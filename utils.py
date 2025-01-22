import cv2
import argparse
import numpy as np
import SimpleITK as sitk 

def convert_to_grayscale(input_path: str, output_path: str) -> None:
    try:
        img = cv2.imread(input_path)
        if img is None:
            raise Exception("Could not read the image")
            
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Save grayscale image
        cv2.imwrite(output_path, gray_img)
        print(f"Successfully converted to grayscale: {input_path} -> {output_path}")
    except Exception as e:
        print(f"Error during conversion: {str(e)}")


def convert_nii_to_png(input_path: str, output_path: str, slice_idx: int = None) -> None:
    try:
        # Load image with ITK
        image = sitk.ReadImage(input_path)
        
        # Convert to numpy array
        img_data = sitk.GetArrayFromImage(image)
        
        # Get total number of slices (ITK uses z,y,x ordering)
        total_slices = img_data.shape[0]
        
        # If no slice index provided, use middle slice
        if slice_idx is None:
            slice_idx = total_slices // 2
            
        # Validate slice index
        if slice_idx < 0 or slice_idx >= total_slices:
            raise ValueError(f"Slice index must be between 0 and {total_slices-1}")
            
        # Extract slice (ITK uses z,y,x ordering)
        slice_data = img_data[slice_idx, :, :]
        
        # Normalize values to 0-255 range
        slice_data = ((slice_data - slice_data.min()) / 
                     (slice_data.max() - slice_data.min()) * 255).astype(np.uint8)
        
        # Save as PNG
        cv2.imwrite(output_path, slice_data)
        print(f"Successfully saved slice {slice_idx} to {output_path}")
        
    except Exception as e:
        print(f"Error during conversion: {str(e)}")

def png_to_jpg(input_path: str, output_path: str, quality: int = 100) -> None:
    try:
        img = cv2.imread(input_path)
        if img is None:
            raise Exception("Could not read the image")
            
        cv2.imwrite(output_path, img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        print(f"Successfully converted: {input_path} -> {output_path}")
    except Exception as e:
        print(f"Error during conversion: {str(e)}")

def jpg_to_png(input_path: str, output_path: str) -> None:
    try:
        img = cv2.imread(input_path)
        if img is None:
            raise Exception("Could not read the image")
            
        cv2.imwrite(output_path, img)
        print(f"Successfully converted: {input_path} -> {output_path}")
    except Exception as e:
        print(f"Error during conversion: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert between image formats')
    parser.add_argument('input', help='Path to input file')
    parser.add_argument('output', help='Path to output file')
    parser.add_argument('--quality', type=int, default=100, help='JPG quality (1-100), default is 95')
    parser.add_argument('--to-jpg', action='store_true', help='Convert from PNG to JPG')
    parser.add_argument('--to-png', action='store_true', help='Convert from JPG to PNG')
    parser.add_argument('--to-gray', action='store_true', help='Convert to grayscale')
    parser.add_argument('--from-nii', action='store_true', help='Convert from NIfTI to PNG')
    parser.add_argument('--slice', type=int, help='Slice index for NIfTI conversion (default is middle slice)')

    args = parser.parse_args()

    if args.from_nii:
        convert_nii_to_png(args.input, args.output, args.slice)
    elif args.to_jpg and args.to_png:
        print("Error: Please select only one conversion direction (--to-jpg OR --to-png)")
    elif args.to_jpg:
        png_to_jpg(args.input, args.output, args.quality)
    elif args.to_png:
        jpg_to_png(args.input, args.output)
    else:
        print("Error: Please specify conversion direction (--to-jpg, --to-png, or --from-nii)") 