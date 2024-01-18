import os
import glob
import nibabel as nib
import numpy as np
from PIL import Image
from PIL import Image
import matplotlib.pyplot as plt

def nifti_to_png(nifti_path, png_path):
    # Load NIfTI file
    nifti_img = nib.load(nifti_path)
    data = nifti_img.get_fdata()

    # Normalize data to 0-255 range (assuming 16-bit or 32-bit data)
    normalized_data = ((data - np.min(data)) / (np.max(data) - np.min(data)) * 255).astype(np.uint8)

    # Convert to PIL Image
    image = Image.fromarray(normalized_data.squeeze())

    # Rotate by 90 degrees to the right
    rotated_image = image.rotate(-90, expand=True)

    flipped_image = rotated_image.transpose(method=Image.FLIP_LEFT_RIGHT)


    # Save as PNG
    flipped_image.save(png_path)
    return flipped_image

def find_jpg_files(folder_path):
    jpg_files = []

    # Recursively walk through the folder tree
    for root, dirs, files in os.walk(folder_path):
        # Use glob to find files with .jpg extension
        jpg_files.extend(glob.glob(os.path.join(root, '*_msp.nii')))

    return jpg_files

input_directory = "/Users/shamistanhuseynov/PycharmProjects/abide_segmentation/abide.cc.bvol.20150118"
output_directory = "/Users/shamistanhuseynov/PycharmProjects/abide_segmentation/dataset/main/"

# Create output directory if it doesn't exist
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

jpg_files = find_jpg_files(input_directory)

if jpg_files:
    first_image_path = jpg_files[0]
    img = nifti_to_png(first_image_path, os.path.join(output_directory, 'first_image.png'))

    # Display the image using matplotlib
    plt.imshow(img, cmap='gray')
    plt.title('First Processed Image')
    plt.axis('off')  # Turn off axis labels
    plt.show()
else:
    print("No NIfTI files found in the specified folder.")