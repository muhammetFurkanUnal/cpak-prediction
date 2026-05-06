"""
Recursively list every file under a directory, writing relative paths to out.txt.

Walks the tree rooted at <root_directory> and writes each file's path
(relative to that root) on its own line. Directories themselves are not
written; only files. Hidden files are included.

Usage:
    python list_files.py <root_directory>

Args:
    root_directory: Directory to scan recursively.

Output:
    out.txt in the current working directory (overwritten if it exists).

Example:
    python list_files.py ./data
    # out.txt:
    #   raw/0.png
    #   raw/1.png
    #   labels/0.json
"""

import os
import sys

def generate_file_list(root_node, output_file="out.txt"):
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            for root, dirs, files in os.walk(root_node):
                for file in files:
                    full_path = os.path.join(root, file)
                    # Calculate the relative path from the root node
                    relative_path = os.path.relpath(full_path, root_node)
                    f.write(relative_path + "\n")
        print(f"Success: {output_file} has been generated.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # Check if the root directory is provided as a terminal argument
    if len(sys.argv) < 2:
        print("Usage: python script_name.py <root_directory>")
    else:
        target_dir = sys.argv[1]
        if os.path.isdir(target_dir):
            generate_file_list(target_dir)
        else:
            print(f"Error: '{target_dir}' is not a valid directory.")