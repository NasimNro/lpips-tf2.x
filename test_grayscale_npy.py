import os
import numpy as np
import tensorflow as tf
from tensorflow.image import resize

from models.lpips_tensorflow import learned_perceptual_metric_model

def load_npy_grayscale(fn):
    """
    Load a grayscale .npy file and convert it to RGB format required by LPIPS
    
    Args:
        fn: Path to .npy file containing grayscale image
        
    Expected input formats:
        - Shape: (height, width) or (height, width, 1)
        - Values: Either [0, 1] or [0, 255] range
        - Any image size (will be resized to 64x64)
        
    Returns:
        tensor of shape (1, 64, 64, 3) with values in [0, 255] range
    """
    # Load the .npy file
    image = np.load(fn)
    
    # If image is 2D (height, width), add channel dimension
    if len(image.shape) == 2:
        image = np.expand_dims(image, axis=-1)
    
    # Resize to 64x64 if needed
    if image.shape[0] != 64 or image.shape[1] != 64:
        image = resize(image, [64, 64])
        
    # Convert to numpy for further processing
    if isinstance(image, tf.Tensor):
        image = image.numpy()
    
    # Ensure values are in range [0, 255]
    if image.max() <= 1.0:
        image = image * 255.0
    
    # Duplicate the grayscale channel to create RGB
    image = np.repeat(image, 3, axis=-1)
    
    # Add batch dimension and convert to float32
    image = np.expand_dims(image, axis=0)
    image = tf.constant(image, dtype=tf.dtypes.float32)
    
    return image

def calculate_lpips(image1_path, image2_path, image_size=64):
    """
    Calculate LPIPS metric between two grayscale images stored as .npy files
    
    Args:
        image1_path: Path to first .npy file
        image2_path: Path to second .npy file
        image_size: Size images will be resized to (default 64)
        
    Returns:
        LPIPS distance as numpy array
    """
    model_dir = './models'
    vgg_ckpt_fn = os.path.join(model_dir, 'vgg', 'exported')
    lin_ckpt_fn = os.path.join(model_dir, 'lin', 'exported')
    
    # Initialize LPIPS model
    lpips = learned_perceptual_metric_model(image_size, vgg_ckpt_fn, lin_ckpt_fn)
    
    # Load and preprocess images
    image1 = load_npy_grayscale(image1_path)
    image2 = load_npy_grayscale(image2_path)
    
    # Calculate metric
    metric = lpips([image1, image2])
    
    return metric.numpy()

# Example usage
if __name__ == "__main__":
    # Example paths - replace with your .npy files
    image1_path = "path/to/your/first_image.npy"
    image2_path = "path/to/your/second_image.npy"
    
    # Calculate LPIPS
    distance = calculate_lpips(image1_path, image2_path)
    print(f"LPIPS distance: {distance[0]:.3f}")

    # Example for processing multiple pairs from a dataset
    def process_dataset(reference_images, distorted_images):
        """
        Process multiple image pairs
        reference_images: list of paths to reference .npy files
        distorted_images: list of paths to distorted .npy files
        """
        all_distances = []
        for ref_path, dist_path in zip(reference_images, distorted_images):
            distance = calculate_lpips(ref_path, dist_path)
            all_distances.append(distance[0])
            print(f"LPIPS distance for {os.path.basename(ref_path)} -> {os.path.basename(dist_path)}: {distance[0]:.3f}")
        
        return np.array(all_distances)

    # Example dataset processing
    # reference_images = ["path/to/ref1.npy", "path/to/ref2.npy"]
    # distorted_images = ["path/to/dist1.npy", "path/to/dist2.npy"]
    # distances = process_dataset(reference_images, distorted_images)
    # print(f"Average LPIPS distance: {distances.mean():.3f}")
