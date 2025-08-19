import cv2
import heapq
from collections import Counter, defaultdict
# import pickle

# Step 1: Load image (grayscale)
img = cv2.imread("gray_image.jpg", cv2.IMREAD_GRAYSCALE)
pixels = img.flatten()

# Step 2: Frequency of pixels
freq = Counter(pixels)

# Step 3: Build Huffman Tree
class Node:
    def __init__(self, symbol, freq):
        self.symbol = symbol
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.freq < other.freq

def build_huffman_tree(freq):
    heap = [Node(sym, fr) for sym, fr in freq.items()]
    heapq.heapify(heap)
    
    while len(heap) > 1:
        n1 = heapq.heappop(heap)
        n2 = heapq.heappop(heap)
        merged = Node(None, n1.freq + n2.freq)
        merged.left, merged.right = n1, n2
        heapq.heappush(heap, merged)
    
    return heap[0]

# Step 4: Generate Huffman Codes
codes = {}

def generate_codes(node, code=""):
    if node is None:
        return
    if node.symbol is not None:
        codes[node.symbol] = code
        return
    generate_codes(node.left, code + "0")
    generate_codes(node.right, code + "1")

root = build_huffman_tree(freq)
generate_codes(root)

# Step 5: Encode image
encoded_img = "".join([codes[p] for p in pixels])

print("Original size (bits):", len(pixels) * 8)
print("Compressed size (bits):", len(encoded_img))
print("Compression Ratio:", round((len(pixels) * 8) / len(encoded_img), 2))
