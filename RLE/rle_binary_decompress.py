from PIL import Image
import numpy as np
import os
import time
import psutil
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error

def rle_decompress(txt_path, output_image_path, original_image_path):
    start_time = time.time()
    
    with open(txt_path, 'r') as file:
        lines = file.readlines()

    height, width = map(int, lines[0].strip().split(' '))
    img_array = np.zeros((height, width), dtype=np.uint8)

    index = 0
    for line in lines[1:]:
        value, count = map(int, line.strip().split(' '))
        fill_value = 255 if value == 1 else 0
        for _ in range(count):
            img_array[index // width][index % width] = fill_value
            index += 1

    img = Image.fromarray(img_array)
    img.save(output_image_path, 'BMP')
    
    end_time = time.time()
    decompression_time = end_time - start_time
    process = psutil.Process(os.getpid())
    memory_usage = process.memory_info().rss / (1024 ** 2)  # Convert bytes to megabytes

    print(f"Decompression Time: {decompression_time:.2f} seconds")
    print(f"Memory Usage: {memory_usage:.2f} MB")

    original_img = np.array(Image.open(original_image_path).convert('L'))
    decompressed_img = np.array(Image.open(output_image_path).convert('L'))

    psnr_value = psnr(original_img, decompressed_img)
    sme_value = mean_squared_error(original_img, decompressed_img)
    ssim_value = ssim(original_img, decompressed_img, data_range=decompressed_img.max() - decompressed_img.min())

    print(f"PSNR: {psnr_value:.2f}")
    print(f"SME: {sme_value:.2f}")
    print(f"SSIM: {ssim_value:.2f}")

def psnr(original, compressed):
    mse = mean_squared_error(original, compressed)
    if mse == 0:
        return 100
    max_pixel = 255.0
    return 20 * np.log10(max_pixel / np.sqrt(mse))

# Example usage
txt_path = '/Users/ahmedalwan/Desktop/FYP/Code/Final/pepper/rle.txt' 
output_image_path = '/Users/ahmedalwan/Desktop/FYP/Code/Final/pepper/recon.bmp' 
original_image_path = '/Users/ahmedalwan/Desktop/FYP/Code/Final/pepper/binary.bmp'
rle_decompress(txt_path, output_image_path, original_image_path)
