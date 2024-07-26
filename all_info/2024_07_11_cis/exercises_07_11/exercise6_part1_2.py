import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.util import random_noise


# Load the fish image
image_path = '/home/eaguiraud/Documents/34269_computer_imaging_and_spectroscopy/2024_07_11_cis/segmentation_key_points/Ex_SegmentationKeypoints/imSeg/Fish.jpg'  
fish_image = cv2.imread(image_path)
fish_image_hsv = cv2.cvtColor(fish_image, cv2.COLOR_BGR2HSV)

# Define color range for segmentation
# lower_color = np.array([lower_hue, lower_saturation, lower_value])  # replace with actual values
# pper_color = np.array([upper_hue, upper_saturation, upper_value])  # replace with actual values
# Hue (H): Represents the color type. It ranges from 0 to 179 in OpenCV. For example, red hues are around 0 or 179, green around 60, and blue around 120. 
# Saturation (S): Represents the intensity of the color. It ranges from 0 to 255, where 0 is white (no color saturation) and 255 is the most saturated color.
# Value (V): Represents the brightness of the color. It ranges from 0 (black) to 255 (brightest).
# /home/eaguiraud/Pictures/Screenshots/opencv_hue.png  ##check here to find the hue value of the color you want to segment

lower_color = np.array([0, 0, 0])  # replace with actual values
upper_color = np.array([20, 255, 255])  # replace with actual values

# Apply thresholding
mask = cv2.inRange(fish_image_hsv, lower_color, upper_color)
segmented_image = cv2.bitwise_and(fish_image, fish_image, mask=mask)

# Display the result for part1_2_a
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Original Fish Image')
plt.imshow(cv2.cvtColor(fish_image, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Segmented Fish Image')
plt.imshow(cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.show()


# Apply Sobel edge detection
sobel_x = cv2.Sobel(segmented_image, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(segmented_image, cv2.CV_64F, 0, 1, ksize=3)
sobel_edge = cv2.magnitude(sobel_x, sobel_y)

# Display the result
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(fish_image, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Sobel Edge Detection')
plt.imshow(sobel_edge, cmap='gray')
plt.axis('off')

plt.show()

# Add Gaussian white noise
noisy_image = random_noise(segmented_image, mode='gaussian', var=0.01)
noisy_image = (255*noisy_image).astype(np.uint8)

# Reapply Sobel edge detection
sobel_x_noisy = cv2.Sobel(noisy_image, cv2.CV_64F, 1, 0, ksize=3)
sobel_y_noisy = cv2.Sobel(noisy_image, cv2.CV_64F, 0, 1, ksize=3)
sobel_edge_noisy = cv2.magnitude(sobel_x_noisy, sobel_y_noisy)

# Display the result for part1_2_b
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




