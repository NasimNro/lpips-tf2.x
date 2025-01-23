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



# Image1 with Distortions
image_fn1 = './imgs/image1/image1.jpg'
image_fn2 = './imgs/image1/image1_SRGAN.jpg'
image1 = load_image(image_fn1)
image2 = load_image(image_fn2)
dist01 = lpips([image1, image2])
print('Image1 Superresolution Distance: {:.3f}'.format(dist01.numpy().item()))

image_fn1 = './imgs/image1/image1.png'
image_fn2 = './imgs/image1/image1_deblurred.png'
image1 = load_image(image_fn1)
image2 = load_image(image_fn2)
dist01 = lpips([image1, image2])
print('Image1 Deblurred Distance: {:.3f}'.format(dist01.numpy().item()))

image_fn1 = './imgs/image1/image1.jpg'
image_fn2 = './imgs/image1/image1_denoise.jpg'
image1 = load_image(image_fn1)
image2 = load_image(image_fn2)
dist01 = lpips([image1, image2])
print('Image1 denoise Distance: {:.3f}'.format(dist01.numpy().item()))

print('--------------------------------')

# Slice 60 with Distortions
image_fn1 = './imgs/brainSlice/brain_slice.jpg'
image_fn2 = './imgs/brainSlice/brain_slice_SRGAN.jpg'
image1 = load_image(image_fn1)
image2 = load_image(image_fn2)
dist01 = lpips([image1, image2])
print('Brain Superresolution Distance: {:.3f}'.format(dist01.numpy().item()))

image_fn1 = './imgs/brainSlice/brain_slice.png'
image_fn2 = './imgs/brainSlice/brain_slice_deblurred.png'
image1 = load_image(image_fn1)
image2 = load_image(image_fn2)
dist01 = lpips([image1, image2])
print('Brain Deblurred Distance: {:.3f}'.format(dist01.numpy().item()))

image_fn1 = './imgs/brainSlice/brain_slice.jpg'
image_fn2 = './imgs/brainSlice/brain_slice_denoise.jpg'
image1 = load_image(image_fn1)
image2 = load_image(image_fn2)
dist01 = lpips([image1, image2])
print('Brain denoise Distance: {:.3f}'.format(dist01.numpy().item()))

print('--------------------------------')

# Slices with each other
image_fn1 = './imgs/brain_slice65.png'
image_fn2 = './imgs/brain_slice66.png'
image1 = load_image(image_fn1)
image2 = load_image(image_fn2)
dist01 = lpips([image1, image2])
print('Distance between slice 65 and 66: {:.3f}'.format(dist01.numpy().item()))

image_fn1 = './imgs/brain_slice65.png'
image_fn2 = './imgs/brain_slice67.png'
image1 = load_image(image_fn1)
image2 = load_image(image_fn2)
dist01 = lpips([image1, image2])
print('Distance between slice 65 and 67: {:.3f}'.format(dist01.numpy().item()))

image_fn1 = './imgs/brain_slice65.png'
image_fn2 = './imgs/brain_slice68.png'
image1 = load_image(image_fn1)
image2 = load_image(image_fn2)
dist01 = lpips([image1, image2])
print('Distance between slice 65 and 68: {:.3f}'.format(dist01.numpy().item()))



