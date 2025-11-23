# src/export_onnx.py

from ultralytics import YOLO
import os

MODEL_PT = "models/yolov8n.pt"
OUTPUT_DIR = "models"

def export_to_onnx():
    print("Loading YOLOv8n model...")
    model = YOLO(MODEL_PT)

    print("Exporting to ONNX...")
    model.export(
        format="onnx",
        opset=12,
        dynamic=True,   # dynamic shapes = works on webcam
        simplify=False
    )

    # Ultralytics saves ONNX next to the .pt file
    print("\nExport complete!")
    print(f"Check folder: {OUTPUT_DIR}")

if __name__ == "__main__":
    export_to_onnx()
