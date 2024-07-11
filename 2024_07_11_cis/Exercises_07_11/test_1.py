import cv2
import matplotlib.pyplot as plt

image_path = '/home/eaguiraud/Documents/34269_computer_imaging_and_spectroscopy/2024_07_11_cis/34269_AWB/Blue_cast.jpg'

# Load the image
image = cv2.imread(image_path)
print('Original Image with Blue Cast:', image.shape)

# Display the original image
image2 = cv2.resize(image, (960, 540))  
cv2.imshow('Original Image with Blue Cast', image2)
cv2.namedWindow("Resized_Window", cv2.WINDOW_NORMAL) 
cv2.waitKey(0)


image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image3 = cv2.resize(image, (960, 540))  
cv2.imshow('Original Image with Blue Cast', image3)
cv2.waitKey(0)

