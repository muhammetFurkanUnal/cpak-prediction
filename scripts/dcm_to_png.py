import os
import argparse
import pydicom
import numpy as np
from PIL import Image

def convert_dcm_to_png(input_dir):
    # Create the output directory (appends _png to the input folder name)
    output_dir = "png"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Directory created: {output_dir}")

    # Scan files in the directory
    for file in os.listdir(input_dir):
        if file.lower().endswith('.dcm'):
            try:
                # Read DICOM file
                dcm_path = os.path.join(input_dir, file)
                ds = pydicom.dcmread(dcm_path)
                
                # Get pixel data and convert to float
                pixel_array = ds.pixel_array.astype(float)
                
                # Normalization: Scale to 0-255 range
                rescaled_image = (np.maximum(pixel_array, 0) / pixel_array.max()) * 255.0
                final_image = Image.fromarray(np.uint8(rescaled_image))
                
                # Create new filename and save
                file_name = os.path.splitext(file)[0] + ".png"
                save_path = os.path.join(output_dir, file_name)
                
                final_image.save(save_path)
                print(f"Success: {file} -> {file_name}")
                
            except Exception as e:
                print(f"Error occurred ({file}): {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert DCM files to PNG.")
    parser.add_argument("folder", help="Name of the folder containing DCM files")
    
    args = parser.parse_args()
    
    if os.path.isdir(args.folder):
        convert_dcm_to_png(args.folder)
    else:
        print(f"Error: '{args.folder}' is not a valid directory path.")