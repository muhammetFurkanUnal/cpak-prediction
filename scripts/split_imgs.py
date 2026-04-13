import os 
import argparse
from split_img import split_image_vertically

def split_images(root_path):
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')
    target_base_dir = "split"

    for root, dirs, files in os.walk(root_path):
        for file in files:
            if file.lower().endswith(valid_extensions):
                # Calculate the relative path from the root_path to the current directory
                rel_path = os.path.relpath(root, root_path)
                
                # Construct the matching directory structure inside the 'split' folder
                save_dir = os.path.join(target_base_dir, rel_path)
                
                full_path = os.path.join(root, file)
                print(f"Processing: {full_path} -> Saving to: {save_dir}")
                
                # Call method with the mirrored directory path
                split_image_vertically(full_path, save_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split all images while preserving directory hierarchy.")
    parser.add_argument("path", help="Root directory path containing images")

    args = parser.parse_args()

    if os.path.isdir(args.path):
        split_images(args.path)
    else:
        print(f"Error: The path '{args.path}' is not a valid directory.")