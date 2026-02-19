import json
import pandas as pd
import argparse
import os

def xml_to_csv(json_path, output_csv, scorer):
    with open(json_path, 'r') as f:
        data = json.load(f)

    categories = data['categories'][0]['keypoints']
    images = {img['id']: img['file_name'] for img in data['images']}
    dlc_data = {}

    for ann in data['annotations']:
        image_name = images[ann['image_id']]
        keypoints = ann['keypoints']
        
        if image_name not in dlc_data:
            dlc_data[image_name] = {}
            
        for i, kp_name in enumerate(categories):
            base_idx = i * 3
            x, y, v = keypoints[base_idx:base_idx + 3]
            
            if v > 0:
                dlc_data[image_name][(scorer, kp_name, 'x')] = x
                dlc_data[image_name][(scorer, kp_name, 'y')] = y
            else:
                dlc_data[image_name][(scorer, kp_name, 'x')] = float('nan')
                dlc_data[image_name][(scorer, kp_name, 'y')] = float('nan')

    df = pd.DataFrame.from_dict(dlc_data, orient='index')
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    df.index.name = 'coords'
    
    df.to_csv(output_csv)
    print(f"Successfully converted {json_path} to {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert COCO keypoints JSON to DeepLabCut CSV.')
    parser.add_argument('--input', type=str, required=True, help='Path to COCO JSON file')
    parser.add_argument('--output', type=str, default='CollectedData_User.csv', help='Path to output CSV')
    parser.add_argument('--scorer', type=str, default='User', help='Scorer name in DLC config')

    args = parser.parse_args()
    xml_to_csv(args.input, args.output, args.scorer)