import cv2
import argparse

def convert_to_grayscale(input_path: str, output_path: str) -> None:
    try:
        img = cv2.imread(input_path)
        if img is None:
            raise Exception("Could not read the image")
            
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(output_path, gray_img)
        print(f"Successfully converted to grayscale: {input_path} -> {output_path}")
    except Exception as e:
        print(f"Error during conversion: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert image to grayscale')
    parser.add_argument('input', help='Input image path')
    parser.add_argument('output', help='Output image path')
    
    args = parser.parse_args()
    convert_to_grayscale(args.input, args.output) 