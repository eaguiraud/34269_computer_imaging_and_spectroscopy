import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.util import random_noise


# Load another image
image_path = '/home/eaguiraud/Documents/34269_computer_imaging_and_spectroscopy/2024_07_11_cis/segmentation_key_points/Ex_SegmentationKeypoints/imSeg/Fish.jpg'  
another_image = cv2.imread(image_path)
another_image_gray = cv2.cvtColor(another_image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur
blurred_image = cv2.GaussianBlur(another_image_gray, (5, 5), 0)

# Apply Otsu's thresholding
_, binary_image = cv2.threshold(blurred_image, 0, 150, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Display the result
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(cv2.cvtColor(another_image, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Segmented Image')
plt.imshow(binary_image, cmap='gray')
plt.axis('off')

plt.show()
