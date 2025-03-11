import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))  # Fügt Root-Verzeichnis zum Path hinzu

import numpy as np
import tensorflow as tf
from dataclasses import dataclass
import cv2
from metrics.psnr import calculate_psnr
from metrics.ssim import calculate_ssim
from metrics.lpips_wrapper import LPIPSMetric
from brainImgs import ImageMetricCalculator
from utils.convert_npz_png import convert_npz_to_png

@dataclass
class ImageMetrics:
    lpips: float
    psnr: float
    ssim: float

def load_image(image_path):
    """Lädt ein Bild und konvertiert es zu float32"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Konnte Bild nicht laden: {image_path}")
    return img.astype(np.float32)

def calculate_metrics_for_pair(calculator, image1, image2):
    """Berechnet Metriken für ein Bildpaar"""
    # Konvertiere zu TF Format für LPIPS
    image1_tf = tf.constant(image1[None, ..., None], dtype=tf.float32)
    image2_tf = tf.constant(image2[None, ..., None], dtype=tf.float32)
    
    lpips = calculator.lpips_metric.calculate_lpips(image1_tf, image2_tf)
    psnr = calculate_psnr(image1, image2)
    ssim = calculate_ssim(image1, image2)
    
    return ImageMetrics(lpips=lpips, psnr=psnr, ssim=ssim)

def process_image_pairs(npz_path, num_slices, output_dir, network_name):
    """Extrahiert Bilder und berechnet Metriken für PNG-Paare"""
    # Initialisierung
    image_size = 256
    model_dir = './models'
    vgg_ckpt_fn = os.path.join(model_dir, 'vgg', 'exported')
    lin_ckpt_fn = os.path.join(model_dir, 'lin', 'exported')
    
    calculator = ImageMetricCalculator(image_size, vgg_ckpt_fn, lin_ckpt_fn)
    
    # Erstelle Ausgabeverzeichnis und extrahiere Bilder
    os.makedirs(output_dir, exist_ok=True)
    
    # Extrahiere die gewünschte Anzahl von Bildpaaren
    for i in range(num_slices):
        convert_npz_to_png(npz_path, i, output_dir)
    
    print(f"\nVerarbeite {num_slices} Bildpaare für Network {network_name}")
    print("=" * 50)
    
    # Lade alle Bilder im Voraus
    originals = []
    predictions = []
    for i in range(num_slices):
        original_path = os.path.join(output_dir, f'original_{i+1}.png')
        prediction_path = os.path.join(output_dir, f'prediction_{i+1}.png')
        
        try:
            originals.append(load_image(original_path))
            predictions.append(load_image(prediction_path))
        except Exception as e:
            print(f"\nFehler beim Laden von Bildpaar {i+1}: {str(e)}")
            return []

    # Vergleiche Original mit Prediction für jedes Paar
    print("\nVergleich Original vs. Prediction:")
    print("=" * 50)
    all_metrics = []
    for i in range(num_slices):
        metrics = calculate_metrics_for_pair(calculator, originals[i], predictions[i])
        all_metrics.append(metrics)
        
        print(f"\nBildpaar {i+1}:")
        print(f"  LPIPS Distance: {metrics.lpips:.3f}")
        print(f"  PSNR: {metrics.psnr:.2f} dB")
        print(f"  SSIM: {metrics.ssim:.3f}")

    # Vergleiche erstes Original mit sich selbst und allen anderen Originalen
    print("\nVergleich Original 1 mit sich selbst und anderen Originalen:")
    print("=" * 50)
    original_comparisons = []
    for i in range(num_slices):
        metrics = calculate_metrics_for_pair(calculator, originals[0], originals[i])
        original_comparisons.append(metrics)
        
        print(f"\nOriginal 1 vs Original {i+1}:")
        print(f"  LPIPS Distance: {metrics.lpips:.3f}")
        print(f"  PSNR: {metrics.psnr:.2f} dB")
        print(f"  SSIM: {metrics.ssim:.3f}")

    # Vergleiche erste Prediction mit sich selbst und allen anderen Predictions
    print("\nVergleich Prediction 1 mit sich selbst und anderen Predictions:")
    print("=" * 50)
    prediction_comparisons = []
    for i in range(num_slices):
        metrics = calculate_metrics_for_pair(calculator, predictions[0], predictions[i])
        prediction_comparisons.append(metrics)
        
        print(f"\nPrediction 1 vs Prediction {i+1}:")
        print(f"  LPIPS Distance: {metrics.lpips:.3f}")
        print(f"  PSNR: {metrics.psnr:.2f} dB")
        print(f"  SSIM: {metrics.ssim:.3f}")

    if all_metrics:
        # Berechne Durchschnittswerte für Original vs. Prediction
        print("\nDurchschnittliche Metriken (Original vs. Prediction):")
        print("=" * 50)
        avg_lpips = np.mean([m.lpips for m in all_metrics])
        avg_psnr = np.mean([m.psnr for m in all_metrics])
        avg_ssim = np.mean([m.ssim for m in all_metrics])
        
        print(f"LPIPS Distance: {avg_lpips:.3f}")
        print(f"PSNR: {avg_psnr:.2f} dB")
        print(f"SSIM: {avg_ssim:.3f}")

        # Berechne Durchschnittswerte für Original 1 vs. andere Originale (ohne Selbstvergleich)
        print("\nDurchschnittliche Metriken (Original 1 vs. andere Originale):")
        print("=" * 50)
        # Überspringe den ersten Eintrag (Selbstvergleich)
        avg_lpips = np.mean([m.lpips for m in original_comparisons[1:]])
        avg_psnr = np.mean([m.psnr for m in original_comparisons[1:]])
        avg_ssim = np.mean([m.ssim for m in original_comparisons[1:]])
        
        print(f"LPIPS Distance: {avg_lpips:.3f}")
        print(f"PSNR: {avg_psnr:.2f} dB")
        print(f"SSIM: {avg_ssim:.3f}")

        # Berechne Durchschnittswerte für Prediction 1 vs. andere Predictions (ohne Selbstvergleich)
        print("\nDurchschnittliche Metriken (Prediction 1 vs. andere Predictions):")
        print("=" * 50)
        # Überspringe den ersten Eintrag (Selbstvergleich)
        avg_lpips = np.mean([m.lpips for m in prediction_comparisons[1:]])
        avg_psnr = np.mean([m.psnr for m in prediction_comparisons[1:]])
        avg_ssim = np.mean([m.ssim for m in prediction_comparisons[1:]])
        
        print(f"LPIPS Distance: {avg_lpips:.3f}")
        print(f"PSNR: {avg_psnr:.2f} dB")
        print(f"SSIM: {avg_ssim:.3f}")
    
    return all_metrics, original_comparisons, prediction_comparisons

def main():
    # Konfiguration
    network_name = '6'
    npz_path = f'./imgs/lungImgs/{network_name}/test_results.npz'
    output_dir = f'./imgs/lungImgs/{network_name}/extracted_images'
    num_slices =  5 # Anzahl der zu verarbeitenden Bildpaare
    
    # Verarbeite die Datei
    process_image_pairs(npz_path, num_slices, output_dir, network_name)

if __name__ == "__main__":
    main() 