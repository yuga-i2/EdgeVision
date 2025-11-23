# src/quantize_onnx.py
import argparse
from onnxruntime.quantization import quantize_dynamic, QuantType

def quantize_model(onnx_input: str, onnx_output: str):
    print(f"Quantizing {onnx_input} -> {onnx_output} (weights-only)...")
    quantize_dynamic(
        model_input=onnx_input,
        model_output=onnx_output,
        weight_type=QuantType.QUInt8  # weights-only quantization compatible with your ORT
    )
    print("Quantization complete (weights-only).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    quantize_model(args.input, args.output)
