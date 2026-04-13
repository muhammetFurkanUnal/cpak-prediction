import os
import argparse
from PIL import Image

def crop_knee_from_leg(input_path, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    
    for filename in os.listdir(input_path):
        if filename.lower().endswith(valid_extensions):
            img_path = os.path.join(input_path, filename)
            with Image.open(img_path) as img:
                width, height = img.size
                
                # 640x1536 senaryosu için diz bölgesi tahmini:
                # Genellikle boyun 1/3'lük orta kısmı (512px - 1024px arası) 
                # diz eklemini kapsar. Bu değerleri ihtiyaca göre güncelleyebilirsiniz.
                top = height * 0.35
                bottom = height * 0.65
                left = 0
                right = width
                
                img_cropped = img.crop((left, top, right, bottom))
                
                save_path = os.path.join(output_path, filename)
                img_cropped.save(save_path)
                print(f"Processed: {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crop knee area from full leg X-rays.")
    parser.add_argument("input", help="Source folder path")
    parser.add_argument("output", help="Destination folder path")
    
    args = parser.parse_args()
    crop_knee_from_leg(args.input, args.output)

# python LLR_to_knee.py <source-path> <target-path>