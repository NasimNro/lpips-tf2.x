import argparse
from utils import *

def main():
    parser = argparse.ArgumentParser(description='Image processing utilities')
    parser.add_argument('--input', help='Path to input file')
    parser.add_argument('--output', help='Path to output file')
    parser.add_argument('--quality', type=int, default=100, help='JPG quality (1-100)')
    parser.add_argument('--to-jpg', action='store_true', help='Convert PNG to JPG')
    parser.add_argument('--to-png', action='store_true', help='Convert JPG to PNG')
    parser.add_argument('--to-gray', action='store_true', help='Convert to grayscale')
    parser.add_argument('--from-nii', action='store_true', help='Convert NIfTI to PNG')
    parser.add_argument('--slice', type=int, help='Slice index for NIfTI conversion')
    parser.add_argument('--save-pair', action='store_true', help='Save image pair from NPZ')
    parser.add_argument('--npz-path', type=str, help='Path to NPZ file')
    parser.add_argument('--index', type=int, help='Index of image pair')
    parser.add_argument('--output-dir', type=str, default='./imgs/Real_Data/saved_images')

    args = parser.parse_args()

    if args.save_pair:
        if not args.npz_path or args.index is None:
            print("Error: --npz-path and --index required")
        else:
            save_image_pair(args.npz_path, args.index, args.output_dir)
    # ... Rest der Logik wie zuvor

if __name__ == "__main__":
    main() 