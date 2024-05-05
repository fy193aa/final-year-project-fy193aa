from PIL import Image
import numpy as np

def convert_to_grayscale_and_save_array(input_image_path, output_image_path, array_output_path):
    img = Image.open(input_image_path)
    grayscale_img = img.convert('L')
    grayscale_img.save(output_image_path)
    print(f"Image successfully converted to grayscale and saved to {output_image_path}.")

    # Convert the grayscale image to an array
    grayscale_array = np.array(grayscale_img)

    # Write the grayscale array to a .txt file
    with open(array_output_path, 'w') as f:
        for row in grayscale_array:
            f.write(' '.join([str(pixel) for pixel in row]) + '\n')

    print(f"Grayscale array saved to {array_output_path}.")

# Image conversion
input_image_path = '/Users/ahmedalwan/Desktop/FYP/Code/Final/pepper/pepper.bmp'
output_image_path = '/Users/ahmedalwan/Desktop/FYP/Code/Final/pepper/gray.bmp'
array_output_path = '/Users/ahmedalwan/Desktop/FYP/Code/Testing Images/grayscale.txt'
convert_to_grayscale_and_save_array(input_image_path, output_image_path, array_output_path)


