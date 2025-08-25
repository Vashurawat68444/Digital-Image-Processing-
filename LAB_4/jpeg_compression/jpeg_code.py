import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# --------------------- Helper Functions ------------------------

def apply_dct(block):
    """2D DCT using OpenCV (block must be float32)."""
    return cv2.dct(block.astype(np.float32))

def apply_idct(block):
    """2D inverse DCT using OpenCV (returns float32)."""
    return cv2.idct(block.astype(np.float32))

def quantize(block, q_matrix):
    """Quantization step"""
    return np.round(block / q_matrix)

def dequantize(block, q_matrix):
    """De-quantization step"""
    return block * q_matrix

# --------------------- JPEG Simulation ------------------------

# Load image and convert to grayscale
image = cv2.imread("fruits.png", cv2.IMREAD_GRAYSCALE)
if image is None:
    raise FileNotFoundError("fruits.png not found. Check the path.")

h, w = image.shape
img_f32 = image.astype(np.float32)

# Pad to multiple of 8
h_pad = (8 - h % 8) % 8
w_pad = (8 - w % 8) % 8
padded = np.pad(img_f32, ((0, h_pad), (0, w_pad)), mode="constant", constant_values=0)
H, W = padded.shape

# Standard JPEG Luminance Quantization Matrix
Q = np.array([
    [16,11,10,16,24,40,51,61],
    [12,12,14,19,26,58,60,55],
    [14,13,16,24,40,57,69,56],
    [14,17,22,29,51,87,80,62],
    [18,22,37,56,68,109,103,77],
    [24,35,55,64,81,104,113,92],
    [49,64,78,87,103,121,120,101],
    [72,92,95,98,112,100,103,99]
], dtype=np.float32)

recon = np.zeros_like(padded, dtype=np.float32)

# Process 8x8 blocks
for i in range(0, H, 8):
    for j in range(0, W, 8):
        block = padded[i:i+8, j:j+8] - 128.0          # shift by -128
        dct_block = apply_dct(block)                   # DCT
        q_block = quantize(dct_block, Q)               # Quantize
        dq_block = dequantize(q_block, Q)              # Dequantize
        idct_block = apply_idct(dq_block) + 128.0      # IDCT + shift back
        recon[i:i+8, j:j+8] = np.clip(idct_block, 0, 255)

# Crop to original size and convert to uint8
jpeg_image = recon[:h, :w].astype(np.uint8)







cv2.imwrite('result.jpg',jpeg_image)

org_size = os.path.getsize("fruits.png")
jpg_size = os.path.getsize("result.jpg")
print("Original File Size on Disk (KB):", org_size / 1024)
print("Jpg File Size on Disk (KB):", jpg_size / 1024)
print("Compression ratio:", org_size/jpg_size)


# --------------------- Display ------------------------
plt.figure(figsize=(10,5))
plt.subplot(1,2,1); plt.title("Original"); plt.imshow(image, cmap='gray'); plt.axis('off')
plt.subplot(1,2,2); plt.title("JPEG Compressed (Simulated)"); plt.imshow(jpeg_image, cmap='gray'); plt.axis('off')
plt.tight_layout(); plt.show()
