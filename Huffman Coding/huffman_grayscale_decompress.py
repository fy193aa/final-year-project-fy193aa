import heapq
import os
import numpy as np
from PIL import Image
import time
import psutil
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error

class HuffmanNode:
    def __init__(self, char, freq):
        self.char = char  # Character (grayscale value) or 'EOF'
        self.freq = freq  # Frequency of the character
        self.left = None  # Left child
        self.right = None  # Right child

    def __lt__(self, other):
        # Properly defined less-than method for comparing two HuffmanNodes
        return self.freq < other.freq

def build_huffman_tree_from_frequencies(frequencies):
    nodes = [HuffmanNode(char, freq) for char, freq in frequencies.items()]
    heapq.heapify(nodes)

    while len(nodes) > 1:
        left = heapq.heappop(nodes)
        right = heapq.heappop(nodes)
        merged = HuffmanNode(None, left.freq + right.freq)
        merged.left = left
        merged.right = right
        heapq.heappush(nodes, merged)

    return nodes[0]

def decode(encoded_data, root):
    decoded_output = []
    current_node = root
    for bit in encoded_data:
        current_node = current_node.left if bit == '0' else current_node.right
        if current_node.char is not None:
            if current_node.char == 'EOF':
                break
            decoded_output.append(current_node.char)
            current_node = root
    return decoded_output

def read_encoded_data(filepath):
    with open(filepath, 'r') as file:
        lines = file.readlines()
    dimensions = tuple(map(int, lines[0].strip().split(',')))
    separator_index = lines.index('-' * 50 + '\n')
    
    # Initialize an empty dictionary for frequencies
    frequencies = {}
    
    # Parse frequency data, skipping lines with non-integer keys or 'Padding'
    for line in lines[1:separator_index]:
        if 'Padding' in line:
            padding = int(line.split()[1])
        else:
            parts = line.split()
            char = parts[0]
            freq = int(parts[1])
            # Handle non-integer keys, specifically the 'EOF' marker
            if char.isdigit():  # Only convert characters that are digits
                char = int(char)  # Convert character key to integer if it is a digit
            frequencies[char] = freq

    encoded_data = ''.join(lines[separator_index + 1:]).replace('\n', '')
    return dimensions, frequencies, encoded_data, padding


def reconstruct_image(dimensions, decoded_pixels):
    height, width = dimensions
    image_array = np.array(decoded_pixels, dtype=np.uint8).reshape((height, width))
    return image_array

def decompress_grayscale_image(input_txt_path, output_image_path, original_image_path):
    start_time = time.time()
    dimensions, frequencies, encoded_data, padding = read_encoded_data(input_txt_path)
    root = build_huffman_tree_from_frequencies(frequencies)
    encoded_data = encoded_data[:-padding]  # Remove padding bits
    decoded_pixels = decode(encoded_data, root)
    image_array = reconstruct_image(dimensions, decoded_pixels)

    img = Image.fromarray(image_array, mode='L')
    img.save(output_image_path)

    end_time = time.time()
    decompression_time = end_time - start_time
    process = psutil.Process(os.getpid())
    memory_usage = process.memory_info().rss / (1024 ** 2)  # Convert bytes to megabytes

    # Load the original and the decompressed image for quality metrics calculation
    original_img = np.array(Image.open(original_image_path).convert('L'))
    decompressed_img = np.array(Image.open(output_image_path).convert('L'))

    mse_value = mean_squared_error(original_img, decompressed_img)
    if mse_value == 0:
        psnr_value = 100
    else:
        psnr_value = 20 * np.log10(255 / np.sqrt(mse_value))
        psnr_value = min(psnr_value, 100)  # Cap PSNR at 100

    ssim_value = ssim(original_img, decompressed_img, data_range=255)

    print(f"Decompression Time: {decompression_time:.2f} seconds")
    print(f"Memory Usage: {memory_usage:.2f} MB")
    print(f"PSNR: {psnr_value:.2f}")
    print(f"SME: {mse_value:.2f}")
    print(f"SSIM: {ssim_value:.2f}")

# Example usage - Update paths as needed
input_txt_path = '/Users/ahmedalwan/Desktop/FYP/Code/Final/pepper/huffman.txt'
output_image_path = '/Users/ahmedalwan/Desktop/FYP/Code/Final/pepper/recon.bmp'
original_image_path = '/Users/ahmedalwan/Desktop/FYP/Code/Final/pepper/gray.bmp'

decompress_grayscale_image(input_txt_path, output_image_path, original_image_path)
