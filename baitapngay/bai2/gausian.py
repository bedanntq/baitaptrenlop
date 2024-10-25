import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Load the image in grayscale mode
image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

save_directory = f'./data/image_tranform'
if not os.path.exists(save_directory):
    os.makedirs(save_directory)


# Sobel operator
sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)  # Gradient in x-direction
sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)  # Gradient in y-direction
sobel_combined = cv2.magnitude(sobel_x, sobel_y)  # Combine both gradients
cv2.imwrite("data/image_tranform/sobel.jpg", sobel_combined)



# Laplacian of Gaussian (LoG) filter
log_kernel = np.array([[0, 0, -1, 0, 0],
                       [0, -1, -2, -1, 0],
                       [-1, -2, 16, -2, -1],
                       [0, -1, -2, -1, 0],
                       [0, 0, -1, 0, 0]])

log_result = cv2.filter2D(image, -1, log_kernel)
cv2.imwrite('data/image_tranform/gaussian.jpg', log_result)

# Display the results
plt.figure(figsize=(10, 8))

plt.subplot(1, 3, 1)
plt.title('Original Image')
plt.imshow(image, cmap='gray')

plt.subplot(1, 3, 2)
plt.title('Sobel Edge Detection')
plt.imshow(sobel_combined, cmap='gray')

plt.subplot(1, 3, 3)
plt.title('Laplacian of Gaussian (LoG)')
plt.imshow(log_result, cmap='gray')

plt.tight_layout()
plt.show()
