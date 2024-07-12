# Load the fish image
image_path = '/home/eaguiraud/Documents/34269_computer_imaging_and_spectroscopy/2024_07_11_cis/segmentation_key_points/Ex_SegmentationKeypoints/imSeg/Fish.jpg'  
fish_image = cv2.imread(image_path)
fish_image_hsv = cv2.cvtColor(fish_image, cv2.COLOR_BGR2HSV)

# Define color range for segmentation
lower_color = np.array([lower_hue, lower_saturation, lower_value])  # replace with actual values
upper_color = np.array([upper_hue, upper_saturation, upper_value])  # replace with actual values

# Apply thresholding
mask = cv2.inRange(fish_image_hsv, lower_color, upper_color)
segmented_image = cv2.bitwise_and(fish_image, fish_image, mask=mask)

# Display the result
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

