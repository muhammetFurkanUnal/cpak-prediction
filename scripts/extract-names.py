import os
import csv
import argparse
import re

def extract_patient_names(folder_path):
    if not os.path.isdir(folder_path):
        print(f"Error: '{folder_path}' is not a valid directory!")
        return

    patient_data = []
    processed_names = set()

    for filename in os.listdir(folder_path):
        if filename.endswith(".dcm"):
            # Boşluk veya nokta karakterlerine göre parçala
            parts = re.split(r'[ .]', filename)
            
            if len(parts) >= 2:
                fname = parts[0]
                lname = parts[1]
                
                full_name = f"{fname} {lname}".upper()
                if full_name not in processed_names:
                    patient_data.append({'FNAME': fname, 'LNAME': lname})
                    processed_names.add(full_name)

    output_file = 'patient_list.csv'
    try:
        with open(output_file, mode='w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['FNAME', 'LNAME']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(patient_data)
        
        print(f"Success: {len(patient_data)} unique patients saved to '{output_file}'.")
        
    except Exception as e:
        print(f"An error occurred: {e}")

def main():
    parser = argparse.ArgumentParser(description="Extract patient names from .dcm files.")
    parser.add_argument("path", help="Directory path containing the DICOM files")
    args = parser.parse_args()

    extract_patient_names(args.path.strip())

if __name__ == "__main__":
    main()