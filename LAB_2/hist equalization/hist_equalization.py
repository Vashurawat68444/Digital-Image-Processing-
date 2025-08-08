import cv2
import numpy as np
import matplotlib.pyplot as plt

# User-defined histogram equalization function
def histogram_equalization(img):
    # Step 1: Calculate histogram
    hist = np.zeros(256, dtype=int)
    for pixel in img.flatten():
        hist[pixel] += 1

    # Step 2: Normalize histogram (PDF)
    pdf = hist / img.size

    # Step 3: Calculate cumulative distribution function (CDF)
    cdf = np.cumsum(pdf)

    # Step 4: Multiply by 255 and round
    equalization_map = np.round(cdf * 255).astype(np.uint8)

    # Step 5: Map original pixel values to equalized values
    equalized_img = equalization_map[img]

    return equalized_img, hist, equalization_map

# Load grayscale image
img = cv2.imread("gray_image.jpg", cv2.IMREAD_GRAYSCALE)

# Apply histogram equalization
equalized_img, hist, eq_map = histogram_equalization(img)

# Plot original and equalized images with histograms
plt.figure(figsize=(10,5))

plt.subplot(2,2,1)
plt.imshow(img, cmap='gray')
plt.title("Original Image")
plt.axis('off')

plt.subplot(2,2,2)
plt.imshow(equalized_img, cmap='gray')
plt.title("Equalized Image")
plt.axis('off')

plt.subplot(2,2,3)
plt.bar(range(256), hist, color='black')
plt.title("Original Histogram")

plt.subplot(2,2,4)
plt.bar(range(256), np.bincount(equalized_img.flatten(), minlength=256), color='black')
plt.title("Equalized Histogram")

plt.tight_layout()
plt.savefig("final.jpg") #savefig always use before show function.
plt.show()

