import cv2
import numpy as np
from collections import Counter

# Step 1: Load Image (grayscale)
img = cv2.imread("gray_image.jpg", cv2.IMREAD_GRAYSCALE)
pixels = img.flatten()

# Step 2: Frequency of pixels
freq = Counter(pixels)
symbols = list(freq.keys())
prob = [f/sum(freq.values()) for f in freq.values()]

# Step 3: Shannon-Fano recursive coding
codes = {}

def shannon_fano(symbols, probs, code=""):
    if len(symbols) == 1:
        codes[symbols[0]] = code
        return
    
    # find split index
    total = sum(probs)
    acc = 0
    split = 0
    for i, p in enumerate(probs):
        acc += p
        if acc >= total/2:
            split = i+1
            break
    
    # Recursive calls
    shannon_fano(symbols[:split], probs[:split], code+"0")
    shannon_fano(symbols[split:], probs[split:], code+"1")

# Step 4: Sort by probability
sorted_items = sorted(freq.items(), key=lambda x: x[1], reverse=True)
symbols, counts = zip(*sorted_items)
probs = [c/sum(counts) for c in counts]

# Step 5: Run Shannon-Fano
shannon_fano(list(symbols), list(probs))

# Step 6: Encode image
encoded_img = "".join([codes[p] for p in pixels])

print("Original size (bits):", len(pixels)*8)
print("Compressed size (bits):", len(encoded_img))
print("Compression Ratio:", round((len(pixels)*8)/len(encoded_img), 2))
