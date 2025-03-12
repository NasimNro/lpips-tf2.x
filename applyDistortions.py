import os
import cv2
from metrics.lpips_wrapper import LPIPSMetric
from metrics.ssim import calculate_ssim
from metrics.psnr import calculate_psnr
from utils.image_distortions import (
    add_gaussian_noise,
    add_salt_pepper_noise,
    add_gaussian_blur,
    add_motion_blur,
)
from dataclasses import dataclass
import numpy as np
import tensorflow as tf

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

def analyze_distortions():
    # Initialize LPIPS metric calculator
    image_size = 256
    model_dir = './models'
    vgg_ckpt_fn = os.path.join(model_dir, 'vgg', 'exported')
    lin_ckpt_fn = os.path.join(model_dir, 'lin', 'exported')
    lpips_metric = LPIPSMetric(image_size, vgg_ckpt_fn, lin_ckpt_fn)

    # Read original image
    input_path = './imgs/brainImgs/brain_slice.png'
    original_image = cv2.imread(input_path)
    if original_image is None:
        raise ValueError(f"Could not read image: {input_path}")

    # Define distortion levels
    distortions = {
        'Gaussian Noise': {
            'levels': [15, 25, 35, 45],
            'function': add_gaussian_noise,
            'param_name': 'sigma'
        },
        'Salt & Pepper': {
            'levels': [0.01, 0.02, 0.03, 0.04],
            'function': add_salt_pepper_noise,
            'param_name': 'prob'
        },
        'Gaussian Blur': {
            'levels': [3, 5, 7, 9],
            'function': add_gaussian_blur,
            'param_name': 'kernel_size'
        },
        'Motion Blur': {
            'levels': [8, 12, 16, 20],
            'function': add_motion_blur,
            'param_name': 'size'
        }
    }

    # Process each distortion type
    for dist_name, dist_info in distortions.items():
        print(f"\n{dist_name}:")
        print("=" * 50)

        for level in dist_info['levels']:
            # Apply distortion
            kwargs = {dist_info['param_name']: level}
            distorted = dist_info['function'](original_image, **kwargs)

            # Calculate metrics
            metrics = calculate_metrics_for_pair(lpips_metric, original_image, distorted)

            # Print results
            print(f"\n{dist_info['param_name']} = {level}:")
            print(f"  LPIPS: {metrics.lpips:.3f}")
            print(f"  PSNR:  {metrics.psnr:.2f} dB")
            print(f"  SSIM:  {metrics.ssim:.3f}")

if __name__ == "__main__":
    analyze_distortions() 