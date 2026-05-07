# python export_onnx.py ./runs/detect/train/weights/best.pt

import sys
import os
from ultralytics import YOLO

def export_model():
    # Check if path argument is provided
    if len(sys.argv) < 2:
        print("Usage: python export_onnx.py <path_to_best.pt>")
        sys.exit(1)

    model_path = sys.argv[1]

    # Check if file exists
    if not os.path.exists(model_path):
        print(f"Error: File '{model_path}' not found.")
        sys.exit(1)

    try:
        # Load the PyTorch model
        print(f"Loading model: {model_path}")
        model = YOLO(model_path)

        # Export to ONNX
        # imgsz: Training image size (640 was used in your training)
        # dynamic: Enables dynamic axes (flexible input sizes)
        # simplify: Optimizes the ONNX graph for better performance
        print("Starting export to ONNX format...")
        path = model.export(format='onnx', imgsz=640, dynamic=True, simplify=True)
        
        print(f"\nSuccess! ONNX model saved to: {path}")

    except Exception as e:
        print(f"An error occurred during export: {e}")
        sys.exit(1)

if __name__ == "__main__":
    export_model()