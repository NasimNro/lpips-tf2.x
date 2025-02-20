import cv2
import numpy as np
from typing import Union

def calculate_ssim(image1: np.ndarray, image2: np.ndarray, 
                  win_size: int = 11, 
                  gaussian_weights: bool = True) -> float:
    """
    Berechnet den Structural Similarity Index (SSIM) zwischen zwei Bildern.
    
    Args:
        image1: Erstes Bild als numpy array
        image2: Zweites Bild als numpy array
        win_size: Fenstergröße für die Gaussian-Filterung (ungerade Zahl)
        gaussian_weights: Ob Gaussian-Gewichtung verwendet werden soll
        
    Returns:
        float: SSIM-Wert zwischen -1 und 1
    """
    if image1.shape != image2.shape:
        raise ValueError("Input images must have the same dimensions")
    
    if len(image1.shape) == 3:  # For color images
        # Convert to luminance
        if image1.shape[2] == 3:
            image1 = cv2.cvtColor(image1.astype('uint8'), cv2.COLOR_RGB2GRAY)
            image2 = cv2.cvtColor(image2.astype('uint8'), cv2.COLOR_RGB2GRAY)
    
    image1 = image1.astype(np.float32)
    image2 = image2.astype(np.float32)
    
    # Constants from the paper
    L = 255  # Dynamic range
    k1 = 0.01
    k2 = 0.03
    c1 = (k1 * L) ** 2
    c2 = (k2 * L) ** 2
    
    # Generate Gaussian kernel if needed
    if gaussian_weights:
        sigma = 1.5
        gaussian_kernel = cv2.getGaussianKernel(win_size, sigma)
        gaussian_window = np.outer(gaussian_kernel, gaussian_kernel.transpose())
    else:
        gaussian_window = np.ones((win_size, win_size))
    
    gaussian_window = gaussian_window / np.sum(gaussian_window)
    
    # Means
    mu1 = cv2.filter2D(image1, -1, gaussian_window)
    mu2 = cv2.filter2D(image2, -1, gaussian_window)
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    # Variances and covariance
    sigma1_sq = cv2.filter2D(image1 ** 2, -1, gaussian_window) - mu1_sq
    sigma2_sq = cv2.filter2D(image2 ** 2, -1, gaussian_window) - mu2_sq
    sigma12 = cv2.filter2D(image1 * image2, -1, gaussian_window) - mu1_mu2
    
    # SSIM calculation
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / \
               ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    
    return float(np.mean(ssim_map)) 