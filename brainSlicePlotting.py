import numpy as np
import SimpleITK as sitk
import cv2
import argparse
import os
import json
import matplotlib.pyplot as plt
from metrics.lpips_wrapper import LPIPSMetric
from metrics.ssim import calculate_ssim
from metrics.psnr import calculate_psnr
import tensorflow as tf
from dataclasses import dataclass
from typing import Dict, List, Tuple

@dataclass
class ImageMetrics:
    lpips: float
    psnr: float
    ssim: float

def normalize_slice(slice_data: np.ndarray) -> np.ndarray:
    """Normalize slice data to 0-255 range."""
    return ((slice_data - slice_data.min()) / 
            (slice_data.max() - slice_data.min()) * 255).astype(np.float32)

def prepare_for_lpips(image: np.ndarray) -> tf.Tensor:
    """Prepare image for LPIPS calculation."""
    # Convert to TF format for LPIPS (already normalized to [0, 255])
    img = tf.constant(image[None, ..., None], dtype=tf.float32)
    # Repeat to create 3 channels
    img = tf.repeat(img, 3, axis=-1)
    return img

def calculate_metrics_for_pair(calculator: LPIPSMetric, image1: np.ndarray, image2: np.ndarray) -> ImageMetrics:
    """Calculate metrics for an image pair"""
    # Convert to TF format for LPIPS
    image1_tf = prepare_for_lpips(image1)
    image2_tf = prepare_for_lpips(image2)
    
    lpips = calculator.calculate_lpips(image1_tf, image2_tf)
    psnr = calculate_psnr(image1, image2)
    ssim = calculate_ssim(image1, image2)
    
    return ImageMetrics(lpips=lpips, psnr=psnr, ssim=ssim)

def plot_results(results: Dict, output_file: str) -> None:
    """Plot metrics in three separate subplots"""
    # Create figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
    fig.suptitle('Metrics vs Distance from Middle Slice')

    # Plot LPIPS
    ax1.plot(results['slice_indices'], results['lpips'], 'b-', linewidth=2)
    ax1.set_ylabel('LPIPS')
    ax1.set_xlabel('Distance from Middle Slice')
    ax1.grid(True)

    # Plot PSNR
    ax2.plot(results['slice_indices'], results['psnr'], 'g-', linewidth=2)
    ax2.set_ylabel('PSNR')
    ax2.set_xlabel('Distance from Middle Slice')
    ax2.grid(True)

    # Plot SSIM
    ax3.plot(results['slice_indices'], results['ssim'], 'r-', linewidth=2)
    ax3.set_ylabel('SSIM')
    ax3.set_xlabel('Distance from Middle Slice')
    ax3.grid(True)

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_file}")
    plt.close()

def analyze_brain_metrics(nii_path: str, metrics_output: str, plot_output: str) -> None:
    try:
        # Initialize LPIPS metric
        image_size = 256
        model_dir = './models'
        vgg_ckpt_fn = os.path.join(model_dir, 'vgg', 'exported')
        lin_ckpt_fn = os.path.join(model_dir, 'lin', 'exported')
        
        lpips_metric = LPIPSMetric(image_size, vgg_ckpt_fn, lin_ckpt_fn)

        # Read the NIfTI file
        image = sitk.ReadImage(nii_path)
        img_data = sitk.GetArrayFromImage(image)
        total_slices = img_data.shape[0]
        middle_slice_idx = total_slices // 2

        # Get and normalize middle slice
        middle_slice = normalize_slice(img_data[middle_slice_idx, :, :])

        # Initialize results dictionary
        results = {
            'slice_indices': [],
            'lpips': [],
            'psnr': [],
            'ssim': []
        }

        print(f"Processing {total_slices} slices...")
        print(f"Using slice {middle_slice_idx} as reference")

        # Calculate metrics for all slices
        for slice_idx in range(total_slices):
            if slice_idx == middle_slice_idx:
                continue

            current_slice = normalize_slice(img_data[slice_idx, :, :])
            metrics = calculate_metrics_for_pair(lpips_metric, current_slice, middle_slice)

            # Store results
            results['slice_indices'].append(slice_idx - middle_slice_idx)  # Relative to middle slice
            results['lpips'].append(float(metrics.lpips))
            results['psnr'].append(float(metrics.psnr))
            results['ssim'].append(float(metrics.ssim))

            print(f"Processed slice {slice_idx}/{total_slices-1}")
            print(f"  LPIPS Distance: {metrics.lpips:.3f}")
            print(f"  PSNR: {metrics.psnr:.2f} dB")
            print(f"  SSIM: {metrics.ssim:.3f}")

        # Calculate average metrics
        avg_lpips = np.mean(results['lpips'])
        avg_psnr = np.mean(results['psnr'])
        avg_ssim = np.mean(results['ssim'])

        print("\nAverage Metrics:")
        print("=" * 50)
        print(f"LPIPS Distance: {avg_lpips:.3f}")
        print(f"PSNR: {avg_psnr:.2f} dB")
        print(f"SSIM: {avg_ssim:.3f}")

        # Save numerical results to JSON file
        with open(metrics_output, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"\nNumerical results saved to {metrics_output}")

        # Create and save the plot
        plot_results(results, plot_output)

    except Exception as e:
        print(f"Error during processing: {str(e)}")

def main(input_path: str = "nii/1/brain.nii", 
         metrics_output: str = "metrics_results.json",
         plot_output: str = "metrics_plot.png") -> None:
    """
    Main function that can be called directly or through command line.
    Args:
        input_path: Path to the input NIfTI file
        metrics_output: Path where to save the JSON results
        plot_output: Path where to save the plot
    """
    analyze_brain_metrics(input_path, metrics_output, plot_output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate and plot metrics between middle slice and all other slices')
    parser.add_argument('input', help='Input NIfTI file path', nargs='?', default="nii/1/brain.nii")
    parser.add_argument('--metrics-output', help='Output JSON file path', default="metrics_results.json")
    parser.add_argument('--plot-output', help='Output plot file path', default="metrics_plot.png")
    
    args = parser.parse_args()
    main(args.input, args.metrics_output, args.plot_output) 