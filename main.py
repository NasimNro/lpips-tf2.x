import os
import numpy as np
import tensorflow as tf
from PIL import Image
from models.lpips_tensorflow import learned_perceptual_metric_model
from utils import convert_nii_to_png


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

image_fn1 = './imgs/image1_ref_gray.png'
image_fn2 = './imgs/srgan_enhanced.jpg'
image1 = load_image(image_fn1)
image2 = load_image(image_fn2)
dist01 = lpips([image1, image2])
print('SRGAN Superresolution Distance: {:.3f}'.format(dist01.numpy().item()))

image_fn1 = './brain_imgs/brain_slice.png'
image_fn2 = './brain_imgs/srgan_enhanced.jpg'
image1 = load_image(image_fn1)
image2 = load_image(image_fn2)
dist01 = lpips([image1, image2])
print('Brain Superresolution Distance: {:.3f}'.format(dist01.numpy().item()))

# Convert middle slice to PNG using ITK

# Or specify a specific slice
# convert_nii_to_png(nii_file_path, output_path, slice_idx=42)

# if brain_slice_path:
#     image_size = 64
#     model_dir = './models'
#     vgg_ckpt_fn = os.path.join(model_dir, 'vgg', 'exported')
#     lin_ckpt_fn = os.path.join(model_dir, 'lin', 'exported')
#     lpips = learned_perceptual_metric_model(image_size, vgg_ckpt_fn, lin_ckpt_fn)

#     # Vergleich von Original und verarbeitetem Bild
#     image_fn1 = brain_slice_path
#     image_fn2 = './imgs/Deblurred.png'
#     image1 = load_image(image_fn1)
#     image2 = load_image(image_fn2)
#     dist01 = lpips([image1, image2])
#     print('\nBrain Slice vs Processed Image comparison:')
#     print('Distance: {:.3f}'.format(dist01.numpy().item()))

#SRGAN Superresolution
# image_fn1 = './imgs/image2_ref.png'
# image_fn2 = './imgs/SRGAN.jpg'
# image1 = load_image(image_fn1)
# image2 = load_image(image_fn2)
# dist01 = lpips([image1, image2])
# # Korrekte Extraktion des einzelnen Wertes aus dem Tensor
# print('SRGAN Superresolution Distance: {:.3f}'.format(dist01.numpy().item()))





# Vergleich von Grayscale und SRGAN enhanced Bildern
# image_fn1 = './imgs/image1_ref_gray.jpg'
# image_fn2 = './imgs/srgan_enhanced.jpg'
# image1 = load_image(image_fn1)
# image2 = load_image(image_fn2)
# dist01 = lpips([image1, image2])
# print('\nGrayscale vs SRGAN Enhanced comparison:')
# print('Distance: {:.3f}'.format(dist01.numpy().item()))


# image_fn1 = './imgs/image1_ref_gray.png'
# image_fn2 = './imgs/Deblurred.png'
# image1 = load_image(image_fn1)
# image2 = load_image(image_fn2)
# dist01 = lpips([image1, image2])
# print('\nGrayscale vs SRGAN Enhanced comparison:')
# print('Distance: {:.3f}'.format(dist01.numpy().item()))
