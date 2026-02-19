import os
import shutil
import argparse

def flatten_directory(root_path):
    # Walk through the directory tree
    # topdown=False is important to process children before parents
    for current_dir, dirs, files in os.walk(root_path, topdown=False):
        for file in files:
            source_path = os.path.join(current_dir, file)
            destination_path = os.path.join(root_path, file)

            # Handle filename collisions
            if os.path.exists(destination_path) and source_path != destination_path:
                name, ext = os.path.splitext(file)
                # Append directory name to distinguish files with same name
                parent_name = os.path.basename(current_dir)
                new_name = f"{name}_{parent_name}{ext}"
                destination_path = os.path.join(root_path, new_name)

            # Move file to root
            if source_path != destination_path:
                shutil.move(source_path, destination_path)
                print(f"Moved: {file} -> {root_path}")

        # Remove the directory if it's empty and not the root
        if current_dir != root_path:
            try:
                os.rmdir(current_dir)
                print(f"Removed empty directory: {current_dir}")
            except OSError:
                print(f"Directory not empty, skipping: {current_dir}")

def main():
    parser = argparse.ArgumentParser(description="Flatten a nested directory structure.")
    parser.add_argument("path", help="Target root directory to flatten")
    args = parser.parse_args()

    if os.path.isdir(args.path):
        flatten_directory(args.path)
    else:
        print(f"Error: {args.path} is not a valid directory.")

if __name__ == "__main__":
    main()