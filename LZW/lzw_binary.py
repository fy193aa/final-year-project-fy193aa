import numpy as np
from PIL import Image
import os
import time
import psutil

def read_image(file_path):
    """Reads an image and converts it to binary format for LZW compression."""
    image = Image.open(file_path).convert('1')  # Convert image to binary (black and white)
    data = np.array(image)
    return data

def lzw_compress(data):
    """Compresses binary data using the LZW algorithm."""
    dictionary = {chr(i): i for i in range(256)}
    dict_size = 256
    p = ""
    compressed = []
    for c in data:
        pc = p + c
        if pc in dictionary:
            p = pc
        else:
            compressed.append(dictionary[p])
            dictionary[pc] = dict_size
            dict_size += 1
            p = c
    if p:
        compressed.append(dictionary[p])
    return compressed

def save_compressed_data(compressed, output_file, dimensions):
    """Saves the compressed data to a file, including image dimensions."""
    with open(output_file, 'w') as file:
        file.write(f"{dimensions[0]},{dimensions[1]}\n")
        for code in compressed:
            file.write(f"{code}\n")

def compress_binary_image(input_file, output_file):
    """Compresses a binary image and saves compressed data, with performance metrics."""
    start_time = time.time()
    data = read_image(input_file)
    flat_data = data.flatten()
    binary_data = ['0' if pixel == 0 else '1' for pixel in flat_data]
    compressed_data = lzw_compress(binary_data)
    save_compressed_data(compressed_data, output_file, data.shape)

    end_time = time.time()
    compression_time = end_time - start_time
    process = psutil.Process(os.getpid())
    memory_usage = process.memory_info().rss / (1024 ** 2)  # Convert bytes to megabytes

    original_size = os.path.getsize(input_file)
    compressed_size = os.path.getsize(output_file)
    original_size_kb = original_size / 1024
    compressed_size_kb = compressed_size / 1024
    compression_ratio = compressed_size / original_size if original_size != 0 else float('inf')


    print(f"Original Size: {original_size_kb:.2f} KB")
    print(f"Compressed Size: {compressed_size_kb:.2f} KB")
    print(f"Compression Ratio: {compression_ratio:.2f}")
    print(f"Compression Time: {compression_time:.2f} seconds")
    print(f"Memory Usage: {memory_usage:.2f} MB")
    print(f"Compression completed. Compressed data saved to {output_file}")

# Example usage
input_file = '/Users/ahmedalwan/Desktop/FYP/Code/Final/pepper/binary.bmp'
output_file = '/Users/ahmedalwan/Desktop/FYP/Code/Final/pepper/lzw.txt'
compress_binary_image(input_file, output_file)
