import cv2
import numpy as np
import argparse
import os
import json
from metrics.lpips_wrapper import LPIPSMetric
from metrics.ssim import calculate_ssim
from metrics.psnr import calculate_psnr
import tensorflow as tf
from dataclasses import dataclass
import matplotlib.pyplot as plt

@dataclass
class ImageMetrics:
    lpips: float
    psnr: float
    ssim: float

def prepare_for_lpips(image: np.ndarray) -> tf.Tensor:
    """Prepare image for LPIPS calculation."""
    # Convert to grayscale if it's RGB
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Normalize to [0, 255] float32
    image = image.astype(np.float32)
    # Add batch and channel dimensions
    img = tf.constant(image[None, ..., None], dtype=tf.float32)
    # Repeat to create 3 channels
    img = tf.repeat(img, 3, axis=-1)
    return img

def calculate_metrics_for_pair(calculator: LPIPSMetric, original: np.ndarray, distorted: np.ndarray) -> ImageMetrics:
    """Calculate metrics between original and distorted images"""
    original_tf = prepare_for_lpips(original)
    distorted_tf = prepare_for_lpips(distorted)
    
    # Convert to grayscale if needed for PSNR and SSIM
    if len(original.shape) == 3:
        original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        distorted_gray = cv2.cvtColor(distorted, cv2.COLOR_BGR2GRAY)
    else:
        original_gray = original
        distorted_gray = distorted
    
    lpips = calculator.calculate_lpips(original_tf, distorted_tf)
    psnr = calculate_psnr(original_gray, distorted_gray)
    ssim = calculate_ssim(original_gray, distorted_gray)
    
    return ImageMetrics(lpips=lpips, psnr=psnr, ssim=ssim)

def add_gaussian_noise(image: np.ndarray, mean=0, sigma=25) -> np.ndarray:
    noise = np.random.normal(mean, sigma, image.shape)
    noisy_image = np.clip(image + noise, 0, 255).astype(np.uint8)
    return noisy_image

def add_salt_pepper_noise(image: np.ndarray, prob=0.02) -> np.ndarray:
    noisy = np.copy(image)
    salt = np.random.random(image.shape) < prob
    pepper = np.random.random(image.shape) < prob
    noisy[salt] = 255
    noisy[pepper] = 0
    return noisy

def add_gaussian_blur(image: np.ndarray, kernel_size=5) -> np.ndarray:
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def add_motion_blur(image: np.ndarray, size=12) -> np.ndarray:
    kernel = np.zeros((size, size))
    kernel[int((size-1)/2), :] = np.ones(size)
    kernel = kernel / size
    return cv2.filter2D(image, -1, kernel)

def plot_metrics(metrics_dict: dict, output_dir: str):
    """Plot metrics for each distortion type"""
    distortion_types = list(metrics_dict.keys())
    metrics_types = ['lpips', 'psnr', 'ssim']
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    fig.suptitle('Metrics vs Distortion Levels')
    
    for idx, metric in enumerate(metrics_types):
        ax = axes[idx]
        for dist_type in distortion_types:
            levels = metrics_dict[dist_type]['levels']
            values = [getattr(m, metric) for m in metrics_dict[dist_type]['metrics']]
            ax.plot(levels, values, marker='o', label=dist_type)
        
        ax.set_ylabel(metric.upper())
        ax.set_xlabel('Distortion Level')
        ax.grid(True)
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

def process_distortions(input_path: str, output_dir: str):
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize LPIPS metric calculator
    image_size = 256
    model_dir = './models'
    vgg_ckpt_fn = os.path.join(model_dir, 'vgg', 'exported')
    lin_ckpt_fn = os.path.join(model_dir, 'lin', 'exported')
    lpips_metric = LPIPSMetric(image_size, vgg_ckpt_fn, lin_ckpt_fn)
    
    # Read original image
    original_image = cv2.imread(input_path)
    
    # Define distortion levels
    gaussian_noise_levels = [15, 25, 35, 45]  # sigma values
    salt_pepper_levels = [0.01, 0.02, 0.03, 0.04]  # probability values
    gaussian_blur_levels = [3, 5, 7, 9]  # kernel sizes
    motion_blur_levels = [8, 12, 16, 20]  # kernel sizes
    
    # Dictionary to store all metrics
    all_metrics = {
        'gaussian_noise': {'levels': gaussian_noise_levels, 'metrics': []},
        'salt_pepper': {'levels': salt_pepper_levels, 'metrics': []},
        'gaussian_blur': {'levels': gaussian_blur_levels, 'metrics': []},
        'motion_blur': {'levels': motion_blur_levels, 'metrics': []}
    }
    
    # Process each distortion type and level
    print("\nProcessing Gaussian Noise:")
    for sigma in gaussian_noise_levels:
        distorted = add_gaussian_noise(original_image, sigma=sigma)
        metrics = calculate_metrics_for_pair(lpips_metric, original_image, distorted)
        all_metrics['gaussian_noise']['metrics'].append(metrics)
        output_path = os.path.join(output_dir, f'gaussian_noise_sigma{sigma}.png')
        cv2.imwrite(output_path, distorted)
        print(f"Sigma {sigma}: LPIPS={metrics.lpips:.3f}, PSNR={metrics.psnr:.2f}, SSIM={metrics.ssim:.3f}")
    
    print("\nProcessing Salt & Pepper Noise:")
    for prob in salt_pepper_levels:
        distorted = add_salt_pepper_noise(original_image, prob=prob)
        metrics = calculate_metrics_for_pair(lpips_metric, original_image, distorted)
        all_metrics['salt_pepper']['metrics'].append(metrics)
        output_path = os.path.join(output_dir, f'salt_pepper_prob{prob}.png')
        cv2.imwrite(output_path, distorted)
        print(f"Probability {prob}: LPIPS={metrics.lpips:.3f}, PSNR={metrics.psnr:.2f}, SSIM={metrics.ssim:.3f}")
    
    print("\nProcessing Gaussian Blur:")
    for kernel_size in gaussian_blur_levels:
        distorted = add_gaussian_blur(original_image, kernel_size=kernel_size)
        metrics = calculate_metrics_for_pair(lpips_metric, original_image, distorted)
        all_metrics['gaussian_blur']['metrics'].append(metrics)
        output_path = os.path.join(output_dir, f'gaussian_blur_kernel{kernel_size}.png')
        cv2.imwrite(output_path, distorted)
        print(f"Kernel Size {kernel_size}: LPIPS={metrics.lpips:.3f}, PSNR={metrics.psnr:.2f}, SSIM={metrics.ssim:.3f}")
    
    print("\nProcessing Motion Blur:")
    for size in motion_blur_levels:
        distorted = add_motion_blur(original_image, size=size)
        metrics = calculate_metrics_for_pair(lpips_metric, original_image, distorted)
        all_metrics['motion_blur']['metrics'].append(metrics)
        output_path = os.path.join(output_dir, f'motion_blur_size{size}.png')
        cv2.imwrite(output_path, distorted)
        print(f"Size {size}: LPIPS={metrics.lpips:.3f}, PSNR={metrics.psnr:.2f}, SSIM={metrics.ssim:.3f}")
    
    # Save metrics to JSON
    metrics_data = {
        dist_type: {
            'levels': info['levels'],
            'metrics': [{'lpips': m.lpips, 'psnr': m.psnr, 'ssim': m.ssim} 
                       for m in info['metrics']]
        }
        for dist_type, info in all_metrics.items()
    }
    
    with open(os.path.join(output_dir, 'distortion_metrics.json'), 'w') as f:
        json.dump(metrics_data, f, indent=4)
    
    # Create plots
    plot_metrics(all_metrics, output_dir)
    print(f"\nResults saved to {output_dir}")

if __name__ == "__main__":
    input_path = './imgs/brainImgs/brain_slice.png'
    output_dir = './imgs/brainImgs/traditional_distortions'
    process_distortions(input_path, output_dir)