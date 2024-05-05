import numpy as np
from PIL import Image
import time
import psutil
import os
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error

class HuffmanNode:
    def __init__(self, char, freq):
        self.char = char
        self.freq = freq
        self.left = None
        self.right = None

def build_huffman_tree_from_frequencies(frequencies):
    nodes = [HuffmanNode(char, freq) for char, freq in frequencies.items()]
    while len(nodes) > 1:
        nodes.sort(key=lambda node: node.freq)
        left = nodes.pop(0)
        right = nodes.pop(0)
        merged = HuffmanNode(None, left.freq + right.freq)
        merged.left = left
        merged.right = right
        nodes.append(merged)
    return nodes[0]

def decode(encoded_data, root):
    decoded_output = []
    current_node = root
    for bit in encoded_data:
        current_node = current_node.left if bit == '0' else current_node.right
        if current_node.char is not None:
            decoded_output.append(current_node.char)
            current_node = root
    return decoded_output

def read_encoded_data(filepath):
    with open(filepath, 'r') as file:
        lines = file.readlines()
    dimensions = tuple(map(int, lines[0].strip().split(',')))
    separator_index = lines.index('-' * 50 + '\n')
    frequencies = {int(line.split()[0]): int(line.split()[1]) for line in lines[1:separator_index]}
    encoded_data = ''.join(lines[separator_index + 1:]).replace('\n', '')
    return dimensions, frequencies, encoded_data

def reconstruct_image(dimensions, decoded_pixels):
    height, width = dimensions
    image_array = np.array(decoded_pixels, dtype=np.uint8).reshape((height, width))
    return image_array

def save_image_pil(image_array, output_image_path):
    img = Image.fromarray(image_array)
    img = img.convert('1')  # Convert the image to 1-bit pixels, black and white
    img.save(output_image_path, 'BMP')  # Save the image in BMP format

def decompress_image(input_txt_path, output_image_path, original_image_path):
    start_time = time.time()
    dimensions, frequencies, encoded_data = read_encoded_data(input_txt_path)
    root = build_huffman_tree_from_frequencies(frequencies)
    decoded_pixels = decode(encoded_data, root)
    image_array = reconstruct_image(dimensions, decoded_pixels)
    
    # Save using PIL to ensure 1-bit depth
    save_image_pil(image_array, output_image_path)
    
    end_time = time.time()
    decompression_time = end_time - start_time
    process = psutil.Process(os.getpid())
    memory_usage = process.memory_info().rss / (1024 ** 2)  # Convert bytes to megabytes

    # Load the original and the decompressed image for quality metrics calculation
    original_img = np.array(Image.open(original_image_path).convert('L'))
    decompressed_img = np.array(Image.open(output_image_path).convert('L'))

    # Calculate PSNR, SME, and SSIM
    mse_value = mean_squared_error(original_img, decompressed_img)
    if mse_value == 0:
        psnr_value = 100  # Cap the PSNR at 100 if MSE is 0
    else:
        psnr_value = 20 * np.log10(255 / np.sqrt(mse_value))
        psnr_value = min(psnr_value, 100)  # Cap the PSNR at 100

    ssim_value = ssim(original_img, decompressed_img, data_range=decompressed_img.max() - decompressed_img.min())

    print(f"Decompression Time: {decompression_time:.2f} seconds")
    print(f"Memory Usage: {memory_usage:.2f} MB")
    print(f"PSNR: {psnr_value:.2f}")
    print(f"SME: {mse_value:.2f}")
    print(f"SSIM: {ssim_value:.2f}")

# Example usage - Update paths as needed
input_txt_path = '/Users/ahmedalwan/Desktop/FYP/Code/Final/pepper/huffman.txt'
output_image_path = '/Users/ahmedalwan/Desktop/FYP/Code/Final/pepper/recon.bmp'
original_image_path = '/Users/ahmedalwan/Desktop/FYP/Code/Final/pepper/binary.bmp'
decompress_image(input_txt_path, output_image_path, original_image_path)
