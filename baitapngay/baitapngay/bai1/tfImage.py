import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread('image_Quunh_NhinSangPhai.png', 0)  # Load as a grayscale image

# 1. Negative Image
negative_image = 255 - image
cv2.imwrite('data/negative_image.jpg', negative_image)  # Lưu ảnh âm tính

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
contrast_image = clahe.apply(image)
cv2.imwrite('data/contrast_image.jpg', contrast_image)  # Lưu ảnh tăng độ tương phản

c = 255 / np.log(1 + np.max(image))
log_image = c * (np.log(1 + image.astype(np.float64)))
log_image = np.array(log_image, dtype=np.uint8)
cv2.imwrite('data/log_image.jpg', log_image)  # Lưu ảnh biến đổi Log

equalized_image = cv2.equalizeHist(image)
cv2.imwrite('data/equalized_image.jpg', equalized_image)  # Lưu ảnh cân bằng histogram

# Display the results
titles = ['Original Image', 'Negative Image', 'Increased Contrast', 'Log Transformation', 'Histogram Equalization']
images = [image, negative_image, contrast_image, log_image, equalized_image]

plt.figure(figsize=(10, 8))
for i in range(5):
    plt.subplot(2, 3, i + 1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')

plt.tight_layout()
plt.show()
