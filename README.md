# EdgeVision

Lightweight example project demonstrating: exporting a YOLOv8 PyTorch model to ONNX, quantizing the ONNX model, running inference with OpenCV, and a small Streamlit demo.

Layout
```
edgevision/
├── README.md
├── environment.txt    # pip freeze (optional)
├── models/
│   ├── yolov8n.pt      # downloaded pretrained PyTorch weights (optional)
│   ├── yolov8n.onnx    # exported ONNX model
│   └── yolov8n_quant.onnx  # quantized ONNX model
├── src/
│   ├── utils.py
│   ├── export_onnx.py
│   ├── quantize_onnx.py
│   ├── run_opencv.py
│   └── run_streamlit.py
└── requirements.txt
```

Quick start

1. Create a venv and install dependencies:

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -r requirements.txt
```

2. Place `yolov8n.pt` in `models/` (or edit the paths in the scripts).

3. Export to ONNX:

```powershell
python src/export_onnx.py --weights models/yolov8n.pt --output models/yolov8n.onnx
```

4. (Optional) Quantize:

```powershell
python src/quantize_onnx.py --input models/yolov8n.onnx --output models/yolov8n_quant.onnx
```

5. Run a quick OpenCV demo on an image or webcam:

```powershell
python src/run_opencv.py --model models/yolov8n.onnx --source tests/data/sample.jpg
```

6. Run the Streamlit demo:

```powershell
streamlit run src/run_streamlit.py
```

Notes
- These scripts try to use the Ultralytics `YOLO` class when available for convenience; otherwise they fall back to basic ONNX runtime usage where possible.
- They are intended as small, maintainable examples — adapt them for production.
