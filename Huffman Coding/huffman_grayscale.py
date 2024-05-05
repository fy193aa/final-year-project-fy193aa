import numpy as np
import cv2
from collections import Counter
import heapq
from PIL import Image
import os
import time
import psutil

class HuffmanNode:
    def __init__(self, char, freq):
        self.char = char
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.freq < other.freq

def calculate_frequencies(image):
    pixels = image.flatten()
    frequencies = Counter(pixels)
    frequencies['EOF'] = 1  # Add EOF marker with a frequency of 1
    return frequencies

def build_huffman_tree(frequencies):
    priority_queue = [HuffmanNode(pixel, freq) for pixel, freq in frequencies.items()]
    heapq.heapify(priority_queue)

    while len(priority_queue) > 1:
        left = heapq.heappop(priority_queue)
        right = heapq.heappop(priority_queue)

        merged = HuffmanNode(None, left.freq + right.freq)
        merged.left = left
        merged.right = right

        heapq.heappush(priority_queue, merged)

    return priority_queue[0]

def build_codes(node, current_code, codes):
    if node is None:
        return

    if node.char is not None:
        codes[node.char] = current_code
        return

    build_codes(node.left, current_code + "0", codes)
    build_codes(node.right, current_code + "1", codes)

def huffman_encoding(image):
    frequencies = calculate_frequencies(image)
    root = build_huffman_tree(frequencies)
    codes = {}
    build_codes(root, "", codes)
    return codes, frequencies

def encode_image(image, codes):
    flattened = image.flatten()
    encoded_output = ''.join([codes[pixel] for pixel in flattened])
    encoded_output += codes['EOF']  # Append the EOF marker
    return encoded_output

def save_encoded_data(filepath, image, codes, frequencies, encoded_data):
    # Calculate padding to make the encoded data a multiple of 8
    padding = (8 - len(encoded_data) % 8) % 8
    encoded_data += '0' * padding  # Add padding bits to the encoded data

    with open(filepath, 'w') as file:
        # Image dimensions
        file.write(f'{image.shape[0]},{image.shape[1]}\n')
        # Frequencies, excluding the EOF marker for clarity in this snippet
        for pixel, freq in frequencies.items():
            file.write(f'{pixel} {freq}\n')
        # Write padding information
        file.write(f'Padding: {padding}\n')
        file.write('-' * 50 + '\n')
        # Encoded data
        file.write(encoded_data)

def compress_grayscale_image(input_image_path, output_txt_path):
    start_time = time.time()
    image = Image.open(input_image_path).convert('L')
    image = np.array(image)
    
    codes, frequencies = huffman_encoding(image)
    encoded_data = encode_image(image, codes)
    save_encoded_data(output_txt_path, image, codes, frequencies, encoded_data)

    end_time = time.time()
    compression_time = end_time - start_time
    process = psutil.Process(os.getpid())
    memory_usage = process.memory_info().rss / (1024 ** 2)  # Convert bytes to megabytes

    original_size = os.path.getsize(input_image_path)
    compressed_size = os.path.getsize(output_txt_path)
    original_size_kb = original_size / 1024
    compressed_size_kb = compressed_size / 1024
    compression_ratio = compressed_size / original_size

    print(f"Original Size: {original_size_kb:.2f} KB")
    print(f"Compressed Size: {compressed_size_kb:.2f} KB")
    print(f"Compression Ratio: {compression_ratio:.2f} (Compressed/Original)")
    print(f"Compression Time: {compression_time:.2f} seconds")
    print(f"Memory Usage: {memory_usage:.2f} MB")

# Example usage - Update paths as needed
input_image_path = '/Users/ahmedalwan/Desktop/FYP/Code/Final/pepper/gray.bmp'  # Path to the grayscale BMP image
output_txt_path = '/Users/ahmedalwan/Desktop/FYP/Code/Final/pepper/huffman.txt'  # Path for saving the Huffman encoded data
compress_grayscale_image(input_image_path, output_txt_path)
