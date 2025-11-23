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


