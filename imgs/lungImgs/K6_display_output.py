# Kocak
"""
In development...
"""

import numpy as np
import matplotlib.pyplot as plt
import glob

def load_data(file_path):
    """Load the data from the npz file."""
    data = np.load(file_path, allow_pickle=True)
    return data['Y'], data['Y_']

def display_images(output_features, predictions, num_images, image_size):
    # Calculate the difference between the original and predicted images
    diff_original_slices = output_features[:num_images] - predictions[:num_images]

    # Normalize based on all values in all 5 difference images
    min_val = np.min(diff_original_slices)
    max_val = np.max(diff_original_slices)

    # Define the color map
    cmap = plt.cm.bwr
    norm = plt.Normalize(vmin=min_val, vmax=max_val)

    fig, axs = plt.subplots(3, num_images, figsize=(15, 15), gridspec_kw={'wspace': 0.1, 'hspace': 0.01})

    for i in range(num_images):
        # Display original image
        axs[0, i].imshow(output_features[i].reshape(image_size, image_size), cmap='gray')
        if i == 0:
            axs[0, i].set_title("Original Images", rotation=90, x=-0.1, y=0.05, fontsize=15)
        axs[0, i].axis('off')

        # Display predicted image
        axs[1, i].imshow(predictions[i].reshape(image_size, image_size), cmap='gray')
        if i == 0:
            axs[1, i].set_title("Synthesized Images", rotation=90, x=-0.1, y=-0.1, fontsize=15)
        axs[1, i].axis('off')

        # Display the difference between the original and predicted images
        im = axs[2, i].imshow(diff_original_slices[i].reshape(image_size, image_size), cmap=cmap, norm=norm)
        if i == 0:
            axs[2, i].set_title("Differences", rotation=90, x=-0.1, y=0.15, fontsize=15)

        # axs[2, i].set_title("Difference Image", rotation=90, x=-0.1, y=0.3)
        axs[2, i].axis('off')

    fig.colorbar(im, ax=axs.ravel().tolist(), orientation='vertical')
    plt.show()

# Load and display single NPZ file
file_path = './imgs/lungImgs/1/test_results.npz'  # Path to your first NPZ file
print(f"\nProcessing: {file_path}")
output_features, predictions = load_data(file_path)
print(f"Total number of image pairs: {len(output_features)}")
display_images(output_features, predictions, 10, 256)
