import cv2
import numpy as np

# Load and resize image
image = cv2.imread("virat.jpg")  # Replace with your image path
image = cv2.resize(image, (600, 600))  # Resize for consistency
cv2.imshow("Original Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply median blur to reduce noise
gray_blur = cv2.medianBlur(gray, 7)

# Use Edge-Preserving Filter for better color simplification
color = cv2.edgePreservingFilter(image, flags=2, sigma_s=150, sigma_r=0.3)

# Detect edges using Adaptive Thresholding (for cartoon-like outlines)
edges = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                              cv2.THRESH_BINARY, blockSize=9, C=2)

# Convert edges to color format
edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

# Reduce colors using K-Means Clustering (creates a cel-shaded effect)
data = np.float32(image).reshape((-1, 3))
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 9  # Number of dominant colors
_, labels, centers = cv2.kmeans(data, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
centers = np.uint8(centers)
segmented_image = centers[labels.flatten()].reshape(image.shape)

# Combine the segmented color image with edges
cartoon = cv2.bitwise_and(segmented_image, edges_colored)



# Display results
cv2.imshow("Cartoonized Image", cartoon)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the output
cv2.imwrite("cartoonized_tv_style.jpg", cartoon)
print("cartoonized image saved successfully!")
