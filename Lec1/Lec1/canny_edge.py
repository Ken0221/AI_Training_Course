import cv2

low_threshold = 50
high_threshold = 100

# Read image
img = cv2.imread('lec2.jpg')

# Convert image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur with kernel size of (3, 3)
blur_gray = cv2.GaussianBlur(gray, (3, 3), 0)

# Perform Canny edge detection
edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

# Save the resulting image
cv2.imwrite('lec2_out.jpg', edges)