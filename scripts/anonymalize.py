import os
import argparse

def rename_files(target_dir, start_number):
    # Check if the directory exists
    if not os.path.isdir(target_dir):
        print(f"Error: '{target_dir}' is not a valid directory.")
        return

    # List all items and filter out directories, keep only files
    files = [f for f in os.listdir(target_dir) if os.path.isfile(os.path.join(target_dir, f))]
    
    # Sort files alphabetically to ensure consistent +0, +1 ordering
    files.sort()

    print(f"Starting rename process in: {target_dir}")
    print(f"Starting index: {start_number}")

    for index, filename in enumerate(files):
        # Get the file extension (e.g., .png, .dcm)
        file_ext = os.path.splitext(filename)[1]
        
        # Calculate the new name: start_number + current_index
        new_name = f"{start_number + index}{file_ext}"
        
        old_path = os.path.join(target_dir, filename)
        new_path = os.path.join(target_dir, new_name)

        try:
            os.rename(old_path, new_path)
            print(f"Renamed: {filename} -> {new_name}")
        except Exception as e:
            print(f"Failed to rename {filename}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rename files in a folder sequentially starting from a given number.")
    parser.add_argument("folder", help="Target directory containing files")
    parser.add_argument("start", type=int, help="The starting number for renaming")

    args = parser.parse_args()
    rename_files(args.folder, args.start)
    

	