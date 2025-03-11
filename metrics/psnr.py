import numpy as np

def calculate_psnr(image1: np.ndarray, image2: np.ndarray) -> float:
    """
    Berechnet den Peak Signal-to-Noise Ratio (PSNR) zwischen zwei Bildern.
    
    Args:
        image1: Erstes Bild als numpy array
        image2: Zweites Bild als numpy array
        
    Returns:
        float: PSNR-Wert in dB
    """
    image1 = image1.astype(np.float32)
    image2 = image2.astype(np.float32)
    
    mse = np.mean((image1 - image2) ** 2)
    if mse == 0:
        return float('inf')  # Unendlicher PSNR bei identischen Bildern
    
    peak = 255.0  # Für 8-Bit Bilder (kann mit np.max(image1) verallgemeinert werden)
    psnr = 10 * np.log10((peak ** 2) / mse)
    
    return psnr
