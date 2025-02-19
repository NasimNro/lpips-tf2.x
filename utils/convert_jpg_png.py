import cv2
import argparse

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
    parser = argparse.ArgumentParser(description='Convert between JPG and PNG')
    parser.add_argument('input', help='Input image path')
    parser.add_argument('output', help='Output image path')
    parser.add_argument('--to-jpg', action='store_true', help='Convert PNG to JPG')
    parser.add_argument('--to-png', action='store_true', help='Convert JPG to PNG')
    parser.add_argument('--quality', type=int, default=100, help='JPG quality (1-100)')
    
    args = parser.parse_args()
    
    if args.to_jpg and args.to_png:
        print("Error: Choose either --to-jpg or --to-png")
    elif args.to_jpg:
        png_to_jpg(args.input, args.output, args.quality)
    elif args.to_png:
        jpg_to_png(args.input, args.output)
    else:
        print("Error: Specify conversion direction (--to-jpg or --to-png)") 