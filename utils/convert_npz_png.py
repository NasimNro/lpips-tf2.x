import os
import argparse
import numpy as np
from PIL import Image

def convert_npz_to_png(npz_path, index, output_dir='./imgs/Real_Data/saved_images'):
    """
    Speichert ein spezifisches Bildpaar aus einer NPZ-Datei.
    
    Args:
        npz_path (str): Pfad zur NPZ-Datei
        index (int): Index des zu speichernden Bildpaares
        output_dir (str): Ausgabeverzeichnis für die Bilder
    """
    data = np.load(npz_path, allow_pickle=True)
    original = data['Y'][index]
    prediction = data['Y_'][index]
    
    original = original.reshape(256, 256)
    prediction = prediction.reshape(256, 256)
    
    original = ((original - original.min()) * (255.0 / (original.max() - original.min()))).astype(np.uint8)
    prediction = ((prediction - prediction.min()) * (255.0 / (prediction.max() - prediction.min()))).astype(np.uint8)
    
    os.makedirs(output_dir, exist_ok=True)
    
    Image.fromarray(original).save(os.path.join(output_dir, f'original_{index+1}.png'))
    Image.fromarray(prediction).save(os.path.join(output_dir, f'prediction_{index+1}.png'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Save image pairs from NPZ file')
    parser.add_argument('--npz-path', type=str, required=True, help='Path to NPZ file')
    parser.add_argument('--index', type=int, required=True, help='Index of image pair')
    parser.add_argument('--output-dir', type=str, default='./imgs/Real_Data/saved_images')
    
    args = parser.parse_args()
    save_image_pair(args.npz_path, args.index, args.output_dir) 