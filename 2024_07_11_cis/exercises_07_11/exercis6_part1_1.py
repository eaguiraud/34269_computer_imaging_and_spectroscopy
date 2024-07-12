import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.util import random_noise

# Load the image
image_path = '/home/eaguiraud/Documents/34269_computer_imaging_and_spectroscopy/2024_07_11_cis/segmentation_key_points/Ex_SegmentationKeypoints/imSeg/Blob.tif'
blob_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Apply Sobel edge detection
sobel_x = cv2.Sobel(blob_image, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(blob_image, cv2.CV_64F, 0, 1, ksize=3)
sobel_edge = cv2.magnitude(sobel_x, sobel_y)

# Display the result
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(blob_image, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Sobel Edge Detection')
plt.imshow(sobel_edge, cmap='gray')
plt.axis('off')

plt.show()


# Add Gaussian white noise
noisy_image = random_noise(blob_image, mode='gaussian', var=0.01)
noisy_image = (255*noisy_image).astype(np.uint8)

# Reapply Sobel edge detection
sobel_x_noisy = cv2.Sobel(noisy_image, cv2.CV_64F, 1, 0, ksize=3)
sobel_y_noisy = cv2.Sobel(noisy_image, cv2.CV_64F, 0, 1, ksize=3)
sobel_edge_noisy = cv2.magnitude(sobel_x_noisy, sobel_y_noisy)

# Display the result
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Noisy Image')
plt.imshow(noisy_image, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Sobel Edge Detection on Noisy Image')
plt.imshow(sobel_edge_noisy, cmap='gray')
plt.axis('off')

plt.show()