import os
import numpy as np
import tensorflow as tf
from PIL import Image
from dataclasses import dataclass
from metrics.psnr import calculate_psnr
from metrics.ssim import calculate_ssim
from metrics.lpips_wrapper import LPIPSMetric

@dataclass
class ImageMetrics:
    lpips: float
    psnr: float
    ssim: float

def load_image(fn: str, return_numpy: bool = False) -> np.ndarray:
    """Lädt und verarbeitet ein Bild"""
    image = Image.open(fn)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize((64, 64), Image.Resampling.LANCZOS)
    image = np.asarray(image)
    if return_numpy:
        return image
    image = np.expand_dims(image, axis=0)
    return tf.constant(image, dtype=tf.dtypes.float32)

class ImageMetricCalculator:
    def __init__(self, image_size: int, vgg_ckpt_fn: str, lin_ckpt_fn: str):
        """Initialisiert den Metric Calculator"""
        self.lpips_metric = LPIPSMetric(image_size, vgg_ckpt_fn, lin_ckpt_fn)
    
    def calculate_metrics(self, image_path1: str, image_path2: str) -> ImageMetrics:
        """
        Berechnet alle Metriken für ein Bildpaar.
        
        Args:
            image_path1: Pfad zum ersten Bild
            image_path2: Pfad zum zweiten Bild
            
        Returns:
            ImageMetrics: Objekt mit allen berechneten Metriken
        """
        # Lade Bilder für verschiedene Metriken
        image1_tf = load_image(image_path1)
        image2_tf = load_image(image_path2)
        image1_np = load_image(image_path1, return_numpy=True)
        image2_np = load_image(image_path2, return_numpy=True)
        
        # Berechne alle Metriken
        lpips = self.lpips_metric.calculate_lpips(image1_tf, image2_tf)
        psnr = calculate_psnr(image1_np, image2_np)
        ssim = calculate_ssim(image1_np, image2_np)
        
        return ImageMetrics(lpips=lpips, psnr=psnr, ssim=ssim)

def print_metrics(description: str, metrics: ImageMetrics):
    """Hilfsfunktion zum Formatieren der Ausgabe"""
    print(f'\n{description}:')
    print(f'  LPIPS Distance: {metrics.lpips:.3f}')
    print(f'  PSNR: {metrics.psnr:.2f} dB')
    print(f'  SSIM: {metrics.ssim:.3f}')

def main():
    # Initialisierung
    image_size = 64
    model_dir = './models'
    vgg_ckpt_fn = os.path.join(model_dir, 'vgg', 'exported')
    lin_ckpt_fn = os.path.join(model_dir, 'lin', 'exported')
    
    calculator = ImageMetricCalculator(image_size, vgg_ckpt_fn, lin_ckpt_fn)
    
    # Definiere Bildpfade
    original_jpg = './imgs/brainSlice/brain_slice.jpg'
    original_png = './imgs/brainSlice/brain_slice.png'
    deblurred = './imgs/brainSlice/brain_slice_deblurred.png'
    denoised = './imgs/brainSlice/brain_slice_denoise.jpg'
    srgan = './imgs/brainSlice/brain_slice_SRGAN.jpg'
    
    # Definiere Slice Pfade
    slice65 = './imgs/brain_slice65.png'  # Original Slice
    slice66 = './imgs/brain_slice66.png'
    slice67 = './imgs/brain_slice67.png'
    slice68 = './imgs/brain_slice68.png'
    
    print('Brain Slice Image Quality Metrics')
    print('=' * 50)
    
    # 1. Vergleich zwischen Original JPG und PNG
    print('\nFormat Comparison:')
    print('-' * 30)
    metrics = calculator.calculate_metrics(original_jpg, original_png)
    print_metrics('JPG vs PNG', metrics)
    
    # 2. Vergleiche mit Original JPG
    print('\nComparisons with Original JPG:')
    print('-' * 30)
    
    # Deblurred vs Original
    metrics = calculator.calculate_metrics(original_jpg, deblurred)
    print_metrics('Original vs Deblurred', metrics)
    
    # Denoised vs Original
    metrics = calculator.calculate_metrics(original_jpg, denoised)
    print_metrics('Original vs Denoised', metrics)
    
    # SRGAN vs Original
    metrics = calculator.calculate_metrics(original_jpg, srgan)
    print_metrics('Original vs SRGAN', metrics)
    
    # 3. Vergleiche mit Original PNG
    print('\nComparisons with Original PNG:')
    print('-' * 30)
    
    # Deblurred vs Original
    metrics = calculator.calculate_metrics(original_png, deblurred)
    print_metrics('Original vs Deblurred', metrics)
    
    # Denoised vs Original
    metrics = calculator.calculate_metrics(original_png, denoised)
    print_metrics('Original vs Denoised', metrics)
    
    # SRGAN vs Original
    metrics = calculator.calculate_metrics(original_png, srgan)
    print_metrics('Original vs SRGAN', metrics)
    
    # 4. Vergleiche zwischen den Slices
    print('\nSlice Comparisons:')
    print('-' * 30)
    print('Comparing with reference slice (65):')
    
    # Slice 66 vs 65
    metrics = calculator.calculate_metrics(slice65, slice66)
    print_metrics('Slice 65 vs 66', metrics)
    
    # Slice 67 vs 65
    metrics = calculator.calculate_metrics(slice65, slice67)
    print_metrics('Slice 65 vs 67', metrics)
    
    # Slice 68 vs 65
    metrics = calculator.calculate_metrics(slice65, slice68)
    print_metrics('Slice 65 vs 68', metrics)

if __name__ == "__main__":
    main()



