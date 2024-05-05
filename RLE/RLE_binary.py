from PIL import Image
import numpy as np
import os
import time
import psutil

def rle_encode(img_array):
    """
    Run-Length Encoding for an image. Assumes img_array contains 0s and 1s as integer values.
    """
    if img_array.dtype != int:
        img_array = img_array.astype(int)

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

def save_rle_to_txt_with_dimensions(rle_data, txt_path, shape):
    """
    Save RLE data to a .txt file with image dimensions included.
    """
    with open(txt_path, 'w') as file:
        file.write(f"{shape[0]} {shape[1]}\n")
        for pixel, count in rle_data:
            # No need to change pixel value, just use as is
            file.write(f"{pixel} {count}\n")

def calculate_metrics(original_path, compressed_path, start_time, end_time):
    """
    Calculate and print compression ratio, compression time, and memory usage.
    """
    original_size = os.path.getsize(original_path)
    compressed_size = os.path.getsize(compressed_path)
    # Convert bytes to kilobytes for a more readable format
    original_size_kb = original_size / 1024
    compressed_size_kb = compressed_size / 1024

    compression_ratio = compressed_size / original_size  # Keep the ratio in bytes for accurate calculation
    compression_time = end_time - start_time
    process = psutil.Process(os.getpid())
    memory_usage = process.memory_info().rss / (1024 ** 2)  # Convert bytes to megabytes

    print(f"Original Size: {original_size_kb:.2f} KB")
    print(f"Compressed Size: {compressed_size_kb:.2f} KB")
    print(f"Compression Ratio: {compression_ratio:.2f} (Compressed/Original)")
    print(f"Compression Time: {compression_time:.2f} seconds")
    print(f"Memory Usage: {memory_usage:.2f} MB")



# Path setup
image_path = '/Users/ahmedalwan/Desktop/FYP/Code/Final/pepper/binary.bmp'
rle_txt_path = '/Users/ahmedalwan/Desktop/FYP/Code/Final/pepper/rle.txt'

# Open and process image
img = Image.open(image_path)
img_array = np.array(img)

# Compression
start_time = time.time()
rle_compressed = rle_encode(img_array)
end_time = time.time()

# Save compressed data
save_rle_to_txt_with_dimensions(rle_compressed, rle_txt_path, img_array.shape[:2])

# Metrics calculation
calculate_metrics(image_path, rle_txt_path, start_time, end_time)

print("RLE data saved to .txt file.")
