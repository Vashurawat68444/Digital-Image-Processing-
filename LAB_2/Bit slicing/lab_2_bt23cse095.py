import cv2
import numpy as np

# Load grayscale image
image = cv2.imread('gray_image.jpg', cv2.IMREAD_GRAYSCALE)

# Function to extract a specific bit plane
def bit_plane_slicing(img, bit):
    # Right shift pixels by 'bit', then get LSB
    return np.uint8((img >> bit) & 1) * 255

# Extract all bit planes (0 to 7)
bit_planes = [bit_plane_slicing(image, i) for i in range(8)]

# Save bit planes
for i, plane in enumerate(bit_planes):
    cv2.imwrite(f'bit_plane_{i}.jpg', plane)

# Display original + all bit planes
cv2.imshow('Original Grayscale', image)
for i in range(8):
    cv2.imshow(f'Bit Plane {i}', bit_planes[i])

cv2.waitKey(0)
cv2.destroyAllWindows()
