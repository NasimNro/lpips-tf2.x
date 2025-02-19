import cv2
import numpy as np

def calculate_ssim(image1: np.ndarray, image2: np.ndarray) -> float:
    """
    Berechnet den Structural Similarity Index (SSIM) zwischen zwei Bildern.
    
    Args:
        image1: Erstes Bild als numpy array
        image2: Zweites Bild als numpy array
        
    Returns:
        float: SSIM-Wert zwischen -1 und 1
    """
    image1 = image1.astype(np.float32)
    image2 = image2.astype(np.float32)
    
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    
    mu1 = cv2.GaussianBlur(image1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(image2, (11, 11), 1.5)
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = cv2.GaussianBlur(image1 ** 2, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(image2 ** 2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(image1 * image2, (11, 11), 1.5) - mu1_mu2
    
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / \
               ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return np.mean(ssim_map) 