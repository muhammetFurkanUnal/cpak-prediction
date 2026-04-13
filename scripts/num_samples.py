import os
import argparse

def count_leaf_files(root_path):
    total_files = 0
    
    # Traverse the entire tree
    for root, dirs, files in os.walk(root_path):
        # files list contains all filenames in the current root
        total_files += len(files)
    
    return total_files

def main():
    parser = argparse.ArgumentParser(description="Count total number of files (leaves) in a dataset.")
    parser.add_argument("path", help="Path to the datasets folder")
    args = parser.parse_args()

    if os.path.isdir(args.path):
        count = count_leaf_files(args.path)
        print(f"Total files found in '{args.path}': {count}")
    else:
        print(f"Error: {args.path} is not a valid directory.")

if __name__ == "__main__":
    main()