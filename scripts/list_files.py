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