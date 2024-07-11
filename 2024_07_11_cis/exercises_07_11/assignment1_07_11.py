import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the image
image_path = 'blue_cast.jpg'
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Display the original image
plt.figure(figsize=(10, 10))
plt.imshow(image)
plt.title('Original Image with Blue Cast')
plt.axis('off')
plt.show()

# Method 1: Gray World Assumption
def gray_world_assumption(image):
    img = image.copy()
    r_avg = np.mean(img[:,:,0])
    g_avg = np.mean(img[:,:,1])
    b_avg = np.mean(img[:,:,2])
    avg = (r_avg + g_avg + b_avg) / 3
    
    img[:,:,0] = np.clip((img[:,:,0] * (avg / r_avg)), 0, 255)
    img[:,:,1] = np.clip((img[:,:,1] * (avg / g_avg)), 0, 255)
    img[:,:,2] = np.clip((img[:,:,2] * (avg / b_avg)), 0, 255)
    
    return img.astype(np.uint8)

# Method 2: White Patch Retinex
def white_patch_retinex(image):
    img = image.copy()
    max_r = np.max(img[:,:,0])
    max_g = np.max(img[:,:,1])
    max_b = np.max(img[:,:,2])
    
    img[:,:,0] = np.clip((img[:,:,0] * (255 / max_r)), 0, 255)
    img[:,:,1] = np.clip((img[:,:,1] * (255 / max_g)), 0, 255)
    img[:,:,2] = np.clip((img[:,:,2] * (255 / max_b)), 0, 255)
    
    return img.astype(np.uint8)

# Apply both methods
image_gray_world = gray_world_assumption(image)
image_white_patch = white_patch_retinex(image)

# Display the results
plt.figure(figsize=(20, 10))

plt.subplot(1, 3, 1)
plt.imshow(image)
plt.title('Original Image with Blue Cast')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(image_gray_world)
plt.title('Gray World Assumption')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(image_white_patch)
plt.title('White Patch Retinex')
plt.axis('off')

plt.show()
