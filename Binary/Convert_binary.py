from PIL import Image
import numpy as np

def convert_to_binary_and_save_array(input_image_path, output_image_path, array_output_path, threshold=128):
    # Load the image
    img = Image.open(input_image_path)

    # Convert image to grayscale
    grayscale_img = img.convert('L')

    # Apply threshold to convert the image to binary
    binary_img = grayscale_img.point(lambda x: 255 if x > threshold else 0, '1')

    # Convert the binary image to an array of 0 and 1
    # PIL's '1' mode image uses 255 for true; convert this to a boolean array, then to int
    binary_array = np.array(binary_img).astype(bool).astype(int)

    # Write the binary array to a .txt file
    with open(array_output_path, 'w') as f:
        for row in binary_array:
            f.write(' '.join([str(pixel) for pixel in row]) + '\n')

    # Save the binary image
    binary_img.save(output_image_path)
    print(f"Conversion done and binary array saved to {array_output_path}.")

# Example usage
input_image_path = '/Users/ahmedalwan/Desktop/FYP/Code/Final/airplane/airplane.png'
output_image_path = '/Users/ahmedalwan/Desktop/FYP/Code/Final/airplane/binary.bmp' 
array_output_path = '/Users/ahmedalwan/Desktop/FYP/Code/Testing Images/array.txt'  
convert_to_binary_and_save_array(input_image_path, output_image_path, array_output_path)
