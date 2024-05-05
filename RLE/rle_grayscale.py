from PIL import Image
import numpy as np
import os
import time
import psutil

def rle_encode_grayscale(img_array):
    """
    Run-Length Encoding for a grayscale image. Assumes img_array contains pixel values ranging from 0 to 255.
    """
    pixels = img_array.flatten()
    rle = []
    prev_pixel = pixels[0]
    count = 1

    for pixel in pixels[1:]:
        if pixel == prev_pixel:
            count += 1
        else:
            rle.append((prev_pixel, count))
            prev_pixel = pixel
            count = 1
    rle.append((prev_pixel, count))
    return rle

def save_rle_to_txt_with_dimensions_grayscale(rle_data, txt_path, shape):
    """
    Save RLE data to a .txt file with image dimensions included, specifically for grayscale images.
    Pixels values and their counts are saved, supporting the full range from 0 to 255.
    """
    with open(txt_path, 'w') as file:
        file.write(f"{shape[0]} {shape[1]}\n")
        for pixel_value, count in rle_data:
            file.write(f"{pixel_value} {count}\n")

def calculate_metrics(original_path, compressed_path, start_time, end_time):
    """
    Calculate and print compression ratio, compression time, and memory usage.
    """
    original_size = os.path.getsize(original_path)
    compressed_size = os.path.getsize(compressed_path)
    original_size_kb = original_size / 1024
    compressed_size_kb = compressed_size / 1024
    compression_ratio = compressed_size / original_size
    compression_time = end_time - start_time
    process = psutil.Process(os.getpid())
    memory_usage = process.memory_info().rss / (1024 ** 2)  # Convert bytes to megabytes

    print(f"Original Size: {original_size_kb:.2f} KB")
    print(f"Compressed Size: {compressed_size_kb:.2f} KB")
    print(f"Compression Ratio: {compression_ratio:.2f} (Compressed/Original)")
    print(f"Compression Time: {compression_time:.2f} seconds")
    print(f"Memory Usage: {memory_usage:.2f} MB")

# Load your grayscale image
image_path = '/Users/ahmedalwan/Desktop/FYP/Code/Final/pepper/gray.bmp'
img = Image.open(image_path)

# Ensure the image is in grayscale mode
if img.mode != 'L':
    img = img.convert('L')

img_array = np.array(img)

# Perform RLE compression
start_time = time.time()
rle_compressed = rle_encode_grayscale(img_array)
end_time = time.time()

# Save RLE compressed data to a .txt file
rle_txt_path = '/Users/ahmedalwan/Desktop/FYP/Code/Final/pepper/rle.txt'
original_shape = img_array.shape[:2]  # This ensures you're only getting the height and width

save_rle_to_txt_with_dimensions_grayscale(rle_compressed, rle_txt_path, original_shape)

# Metrics calculation
calculate_metrics(image_path, rle_txt_path, start_time, end_time)

print("RLE data saved to .txt file.")
