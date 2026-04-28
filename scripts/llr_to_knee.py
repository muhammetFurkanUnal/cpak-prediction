import cv2
import os
import glob
import argparse

def crop_with_template(image_path, template, output_dir):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to load: {image_path}")
        return
        
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
    _, _, _, max_loc = cv2.minMaxLoc(res)
    
    top_left = max_loc
    h, w = template.shape
    bottom_right = (top_left[0] + w, top_left[1] + h)
    
    cropped_img = img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
    
    base_name = os.path.basename(image_path)
    save_path = os.path.join(output_dir, "matched_" + base_name)
    cv2.imwrite(save_path, cropped_img)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("template")
    parser.add_argument("output")
    args = parser.parse_args()

    if not os.path.isdir(args.input):
        print(f"Error: Input directory not found -> {args.input}")
        return

    os.makedirs(args.output, exist_ok=True)
    
    template = cv2.imread(args.template, 0)
    if template is None:
        print(f"Error: Template not found -> {args.template}")
        return

    image_paths = []
    extensions = ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG')
    
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(args.input, ext)))
        
    print(f"Found {len(image_paths)} images in {args.input}")

    if len(image_paths) == 0:
        return

    for path in image_paths:
        crop_with_template(path, template, args.output)
        
    print("Processing complete.")

if __name__ == "__main__":
    main()