import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

def show_image(image, figsize=(5,5)):
    plt.figure(figsize=figsize)
    plt.imshow(image, cmap='gray')
    plt.axis('off')  # Eksenleri gizle
    plt.show() # Pencereyi kapattığın an kod bir sonraki satıra geçer

def load_image(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at: {image_path}")
    
    # Load as is (BGR color or Grayscale depending on file)
    # Using IMREAD_UNCHANGED ensures no info is lost initially
    image = cv2.imread(image_path)
    
    if image is None:
        raise ValueError(f"Could not decode image at: {image_path}")
        
    return image

def save_ndarray_as_image(image, output_path):
    # Eğer resim float32 formatındaysa (0-1 arası), 0-255 arasına çek
    if image.dtype == np.float32:
        # Z-score uygulandıysa önce değerleri 0-255 arasına normalize et
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    
    # Veri tipini uint8'e dönüştür (Resim formatları için zorunlu)
    image_uint8 = image.astype(np.uint8)
    
    # Kaydet
    success = cv2.imwrite(output_path, image_uint8)
    
    if not success:
        print(f"Failed to save image to: {output_path}")

def apply_fast_resize(image, target_max_dim=1024):
    h, w = image.shape[:2]
    scale = target_max_dim / max(h, w)
    if scale >= 1: return image
    return cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

def apply_grayscale(image):
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image

def apply_clahe(image, clip_limit=2.0, grid_size=(8, 8)):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    return clahe.apply(image)

def apply_letterbox(image, target_size=(640, 640)):
    h, w = image.shape[:2]
    tw, th = target_size
    scale = min(tw / w, th / h)
    nw, nh = int(w * scale), int(h * scale)
    
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    
    canvas = np.zeros((th, tw), dtype=np.uint8)
    dx = (tw - nw) // 2
    dy = (th - nh) // 2
    canvas[dy:dy+nh, dx:dx+nw] = resized
    return canvas

def apply_z_score_normalization(image):
    mean, std = image.mean(), image.std()
    return (image - mean) / (std + 1e-7)

def apply_bilateral_filter(image, d=5, sigma_color=75, sigma_space=75):
    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)

def apply_denoising(image, kernel_size=3):
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def apply_normalization(image):
    return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)

def apply_to_tensor(image):
    # Do this ONLY at the very end to save RAM
    return image.astype(np.float32) / 255.0


def apply_crop_white_padding(image, threshold=220):
    # Ensure we are working with a grayscale mask for coordinates
    if len(image.shape) == 3:
        # If BGR, convert to gray for masking
        temp_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mask = temp_gray < threshold
    else:
        mask = image < threshold

    coords = np.argwhere(mask)
    
    if coords.size == 0:
        return image
        
    # Take only y and x (first two columns) to avoid unpack errors with RGB
    y0, x0 = coords[:, :2].min(axis=0)
    y1, x1 = coords[:, :2].max(axis=0)
    
    # Add a small margin (e.g., 10px) so we don't cut too close to the bone
    margin = 1
    y0 = max(0, y0 - margin)
    x0 = max(0, x0 - margin)
    y1 = min(image.shape[0], y1 + margin)
    x1 = min(image.shape[1], x1 + margin)
    
    return image[y0:y1+1, x0:x1+1]

def apply_black_letterbox(image, target_size=(640, 640)):
    h, w = image.shape[:2]
    tw, th = target_size
    scale = min(tw / w, th / h)
    nw, nh = int(w * scale), int(h * scale)
    
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)
    
    # Using black (0) for padding is standard for deep learning
    canvas = np.zeros((th, tw), dtype=np.uint8)
    dx = (tw - nw) // 2
    dy = (th - nh) // 2
    canvas[dy:dy+nh, dx:dx+nw] = resized
    
    return canvas