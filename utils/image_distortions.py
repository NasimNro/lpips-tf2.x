import cv2
import numpy as np



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
