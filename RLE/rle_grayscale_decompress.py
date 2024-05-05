from PIL import Image
import numpy as np
import os
import time
import psutil
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error

def rle_decompress_grayscale(txt_path, output_image_path, original_image_path):
    """
    Decompress RLE data from a .txt file for a grayscale image and reconstruct
    the original image, ensuring it matches the original BMP in appearance and file size,
    and calculating performance and quality metrics.
    """
    start_time = time.time()

    # Open and read the RLE compressed data
    with open(txt_path, 'r') as file:
        lines = file.readlines()

    # Extract image dimensions
    height, width = map(int, lines[0].strip().split(' '))
    
    # Initialize an empty array to hold the decompressed pixel data
    img_array = np.zeros((height, width), dtype=np.uint8)

    # Process the RLE data
    index = 0
    for line in lines[1:]:
        value, count = map(int, line.strip().split(' '))
        for _ in range(count):
            img_array[index // width][index % width] = value
            index += 1

    # Convert the numpy array to a PIL Image object in L mode (grayscale)
    img = Image.fromarray(img_array, mode='L')

    # Save the image in BMP format
    img.save(output_image_path, 'BMP')

    end_time = time.time()
    decompression_time = end_time - start_time
    process = psutil.Process(os.getpid())
    memory_usage = process.memory_info().rss / (1024 ** 2)  # Convert bytes to megabytes

    print(f"Decompression Time: {decompression_time:.2f} seconds")
    print(f"Memory Usage: {memory_usage:.2f} MB")

    # Load the original and the decompressed image for quality metrics calculation
    original_img = np.array(Image.open(original_image_path).convert('L'))
    decompressed_img = np.array(Image.open(output_image_path).convert('L'))

    # Calculate PSNR, MSE, and SSIM
    mse_value = mean_squared_error(original_img, decompressed_img)
    if mse_value == 0:
        psnr_value = 100  # Cap the PSNR at 100 if MSE is 0
    else:
        psnr_value = 20 * np.log10(255 / np.sqrt(mse_value))
        psnr_value = min(psnr_value, 100)  # Cap the PSNR at 100

    ssim_value = ssim(original_img, decompressed_img, data_range=decompressed_img.max() - decompressed_img.min())

    print(f"PSNR: {psnr_value:.2f}")
    print(f"SME: {mse_value:.2f}")
    print(f"SSIM: {ssim_value:.2f}")

# Example usage
txt_path = '/Users/ahmedalwan/Desktop/FYP/Code/Final/pepper/rle.txt'
output_image_path = '/Users/ahmedalwan/Desktop/FYP/Code/Final/pepper/recon.bmp'
original_image_path = '/Users/ahmedalwan/Desktop/FYP/Code/Final/pepper/gray.bmp'
rle_decompress_grayscale(txt_path, output_image_path, original_image_path)
