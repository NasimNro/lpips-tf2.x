import os
import numpy as np
import tensorflow as tf
from PIL import Image
from glob import glob

from models.lpips_tensorflow import learned_perceptual_metric_model


def load_image(fn):
    image = Image.open(fn)
    # Ensure RGB format
    if image.mode != 'RGB':
        image = image.convert('RGB')
    # Resize to 64x64 pixels
    image = image.resize((64, 64), Image.Resampling.LANCZOS)
    image = np.asarray(image)
    image = np.expand_dims(image, axis=0)
    image = tf.constant(image, dtype=tf.dtypes.float32)
    return image


image_size = 64
model_dir = './models'
vgg_ckpt_fn = os.path.join(model_dir, 'vgg', 'exported')
lin_ckpt_fn = os.path.join(model_dir, 'lin', 'exported')
lpips = learned_perceptual_metric_model(image_size, vgg_ckpt_fn, lin_ckpt_fn)


#SRGAN Superresolution
image_fn1 = './imgs/image2_ref.png'
image_fn2 = './imgs/SRGAN.jpg'
image1 = load_image(image_fn1)
image2 = load_image(image_fn2)
dist01 = lpips([image1, image2])
# Korrekte Extraktion des einzelnen Wertes aus dem Tensor
print('SRGAN Superresolution Distance: {:.3f}'.format(dist01.numpy().item()))


#Deblurring
image_fn1 = './imgs/image2_ref.png'
image_fn2 = './imgs/Deblur.png'
image1 = load_image(image_fn1)
image2 = load_image(image_fn2)
dist01 = lpips([image1, image2])
# Korrekte Extraktion des einzelnen Wertes aus dem Tensor
print('Deblurring Distance: {:.3f}'.format(dist01.numpy().item()))


#FGSM
image_fn1 = './imgs/image2_ref.png'
image_fn2 = './imgs/FGSM.png'
image1 = load_image(image_fn1)
image2 = load_image(image_fn2)
dist01 = lpips([image1, image2])
# Korrekte Extraktion des einzelnen Wertes aus dem Tensor
print('FGSM Distance: {:.3f}'.format(dist01.numpy().item()))

#Colorization
image_fn1 = './imgs/image2_ref.png'
image_fn2 = './imgs/Colorization.jpg'
image1 = load_image(image_fn1)
image2 = load_image(image_fn2)
dist01 = lpips([image1, image2])
# Korrekte Extraktion des einzelnen Wertes aus dem Tensor
print('Colorization Distance: {:.3f}'.format(dist01.numpy().item()))

#Colorization
image_fn1 = './imgs/image2_ref.png'
image_fn2 = './imgs/image2_ref.png'
image1 = load_image(image_fn1)
image2 = load_image(image_fn2)
dist01 = lpips([image1, image2])
# Korrekte Extraktion des einzelnen Wertes aus dem Tensor
print('Same Image Distance: {:.3f}'.format(dist01.numpy().item()))

# Get all reference images
ref_images = glob('./imgs/*ref.png')

for ref_path in ref_images:
    try:
        # Get the base name of the image (e.g., 'ex' from 'ex_ref.png')
        base_name = os.path.basename(ref_path).split('_ref')[0]
        
        # Construct paths for p0 and p1
        p0_path = os.path.join('./imgs', f'{base_name}_p0.png')
        p1_path = os.path.join('./imgs', f'{base_name}_p1.png')
        
        print(f'\nProcessing image set: {base_name}')
        
        # Load images
        ref_img = load_image(ref_path)
        p0_img = load_image(p0_path)
        p1_img = load_image(p1_path)
        
        # Create batches for comparison
        batch_ref = tf.concat([ref_img, ref_img], axis=0)
        batch_inp = tf.concat([p0_img, p1_img], axis=0)
        
        # Calculate metrics
        metric = lpips([batch_ref, batch_inp])
        
        print(f'ref shape: {batch_ref.shape}')
        print(f'inp shape: {batch_inp.shape}')
        print(f'lpips metric shape: {metric.shape}')
        print(f'ref <-> p0: {metric[0]:.3f}')
        print(f'ref <-> p1: {metric[1]:.3f}')
    
    except Exception as e:
        print(f"Error processing {base_name}: {str(e)}")
        continue
