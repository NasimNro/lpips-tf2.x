import tensorflow as tf
from models.lpips_tensorflow import learned_perceptual_metric_model

class LPIPSMetric:
    def __init__(self, image_size: int, vgg_ckpt_fn: str, lin_ckpt_fn: str):
        """
        Initialisiert das LPIPS-Modell.
        
        Args:
            image_size: Größe der Eingabebilder
            vgg_ckpt_fn: Pfad zum VGG-Checkpoint
            lin_ckpt_fn: Pfad zum Linear-Layer-Checkpoint
        """
        self.model = learned_perceptual_metric_model(image_size, vgg_ckpt_fn, lin_ckpt_fn)
    
    def calculate_lpips(self, image1: tf.Tensor, image2: tf.Tensor) -> float:
        """
        Berechnet die LPIPS-Distanz zwischen zwei Bildern.
        
        Args:
            image1: Erstes Bild als TensorFlow Tensor
            image2: Zweites Bild als TensorFlow Tensor
            
        Returns:
            float: LPIPS-Distanz
        """
        dist = self.model([image1, image2])
        return dist.numpy().item() 