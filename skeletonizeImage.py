import cv2
import numpy as np
from skimage.morphology import skeletonize
from skimage.util import invert
import matplotlib.pyplot as plt

def skeletonize_image(image_path, output_path=None, show=False):
    # Read the image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Convert to binary image (thresholding)
    _, binary = cv2.threshold(img, 200, 3, cv2.THRESH_BINARY_INV)

    # Apply skeletonization
    skeleton = skeletonize(binary)

    # Convert back to 0-255 image for saving
    skeleton_uint8 = (invert(skeleton) * 255).astype(np.uint8)

    if output_path:
        cv2.imwrite(output_path, skeleton_uint8)

    if show:
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.title("Original")
        plt.imshow(img, cmap='gray')

        plt.subplot(1, 2, 2)
        plt.title("Skeleton")
        plt.imshow(skeleton_uint8, cmap='gray')
        plt.show()

    return skeleton_uint8

# Example usage
skeleton = skeletonize_image("1407.png", "1407_skeleton.jpg", show=True)
