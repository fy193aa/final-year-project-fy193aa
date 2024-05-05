import numpy as np
import os
from PIL import Image
import time
import psutil
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error

def read_compressed_data(input_file):
    with open(input_file, 'r') as file:
        lines = file.readlines()
        dimensions = tuple(map(int, lines[0].strip().split(',')))
        compressed_data = [int(line.strip()) for line in lines[1:]]
    return compressed_data, dimensions

def lzw_decompress(compressed):
    dictionary = {i: chr(i) for i in range(256)}
    result = []
    
    w = chr(compressed.pop(0))
    result.append(w)
    
    for k in compressed:
        if k in dictionary:
            entry = dictionary[k]
        else:
            entry = w + w[0]
        result.append(entry)
        
        dictionary[len(dictionary)] = w + entry[0]
        w = entry
    
    return ''.join(result)

def reconstruct_image(data, dimensions):
    pixels = [ord(pixel) for pixel in data]
    image_array = np.array(pixels).reshape(dimensions)
    image = Image.fromarray(image_array.astype('uint8'), 'L')
    return image

def decompress_grayscale_image(input_file, output_file, original_image_path):
    start_time = time.time()
    compressed_data, dimensions = read_compressed_data(input_file)
    decompressed_data = lzw_decompress(compressed_data)
    image = reconstruct_image(decompressed_data, dimensions)
    image.save(output_file)

    end_time = time.time()
    decompression_time = end_time - start_time
    process = psutil.Process(os.getpid())
    memory_usage = process.memory_info().rss / (1024 ** 2)  # Convert bytes to megabytes

    # Load the original and the decompressed image for quality metrics calculation
    original_img = np.array(Image.open(original_image_path).convert('L'))
    decompressed_img = np.array(Image.open(output_file).convert('L'))

    mse_value = mean_squared_error(original_img, decompressed_img)
    psnr_value = 20 * np.log10(255 / np.sqrt(mse_value)) if mse_value != 0 else float('inf')
    psnr_value = min(psnr_value, 100)  # Cap PSNR at 100
    ssim_value = ssim(original_img, decompressed_img, data_range=255)

    print(f"Decompression Time: {decompression_time:.2f} seconds")
    print(f"Memory Usage: {memory_usage:.2f} MB")
    print(f"PSNR: {psnr_value:.2f}")
    print(f"SME: {mse_value:.2f}")
    print(f"SSIM: {ssim_value:.2f}")
    print(f"Decompression completed. Image saved to {output_file}")

# Example usage - Update paths as needed
input_file = '/Users/ahmedalwan/Desktop/FYP/Code/Final/barbara/lzw.txt'
output_file = '/Users/ahmedalwan/Desktop/FYP/Code/Final/barbara/recon.bmp'
original_image_path = '/Users/ahmedalwan/Desktop/FYP/Code/Final/barbara/gray.bmp'
decompress_grayscale_image(input_file, output_file, original_image_path)
