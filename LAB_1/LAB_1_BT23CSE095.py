import cv2
import numpy as np


# ---------------------------------- FUNCTIONS BLOCK ---------------------------------- #
def rgb_to_grayscale(image):
    """
    Convert RGB image to grayscale using weighted formula based on human perception.
    """
    B = image[:, :, 0]
    G = image[:, :, 1]
    R = image[:, :, 2]

    # Apply grayscale conversion formula
    gray = 0.114 * B + 0.587 * G + 0.299 * R

    return gray.astype(np.uint8)


def rgb_to_bw(image, threshold=127):
    """
    Convert RGB image to black and white using manual thresholding.
    """
    B = image[:, :, 0]
    G = image[:, :, 1]
    R = image[:, :, 2]

    # Grayscale conversion
    gray = 0.114 * B + 0.587 * G + 0.299 * R
    gray = gray.astype(np.uint8)

    # Apply binary threshold
    bw = np.zeros_like(gray)
    bw[gray > threshold] = 255
    bw[gray <= threshold] = 0

    return bw


def extract_single_color(image, channel):
    """
    Return an image with only one color channel visible.
    channel: 'R', 'G', or 'B'
    """
    zeros = np.zeros_like(image[:, :, 0])
    if channel == 'R':
        return cv2.merge([zeros, zeros, image[:, :, 2]])
    elif channel == 'G':
        return cv2.merge([zeros, image[:, :, 1], zeros])
    elif channel == 'B':
        return cv2.merge([image[:, :, 0], zeros, zeros])
    else:
        raise ValueError("Channel must be 'R', 'G', or 'B'")


# ---------------------------------- MAIN BLOCK ---------------------------------- #

# Load the RGB image
image = cv2.imread('fruits.png')  # Replace with your image path

# Convert to grayscale
gray_image = rgb_to_grayscale(image)
cv2.imwrite('gray_image.jpg', gray_image)

# Convert to Black & White
bw_image = rgb_to_bw(image, threshold=127)
cv2.imwrite('black_white.jpg', bw_image)

# Extract individual color channels
red_image = extract_single_color(image, 'R')
green_image = extract_single_color(image, 'G')
blue_image = extract_single_color(image, 'B')

# Save color channel images
cv2.imwrite('red.jpg', red_image)
cv2.imwrite('green.jpg', green_image)
cv2.imwrite('blue.jpg', blue_image)

# Display all results
cv2.imshow('Original Image', image)
cv2.imshow('Grayscale Image', gray_image)
cv2.imshow('Black and White', bw_image)
cv2.imshow('Red Channel', red_image)
cv2.imshow('Green Channel', green_image)
cv2.imshow('Blue Channel', blue_image)

cv2.waitKey(0)
cv2.destroyAllWindows()