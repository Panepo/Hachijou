"""
Convert YOLOv11 PyTorch model to ONNX format
"""
import argparse
from pathlib import Path
from ultralytics import YOLO


def convert_to_onnx(model_path, imgsz=640, simplify=True):
    """
    Convert YOLO model to ONNX format.

    Args:
        model_path (str): Path to the .pt model file
        imgsz (int): Input image size (default: 640)
        simplify (bool): Simplify ONNX model (default: True)
    """
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    print(f"Loading model: {model_path}")
    model = YOLO(str(model_path))

    print(f"Converting to ONNX format (image size: {imgsz})...")
    model.export(
        format='onnx',
        imgsz=imgsz,
        simplify=simplify,
        opset=12,  # ONNX opset version
        dynamic=False  # Static input shape for better compatibility
    )

    # The exported file will be in the same directory as the input
    output_path = model_path.parent / f"{model_path.stem}.onnx"
    print(f"✓ Conversion complete!")
    print(f"  Output: {output_path}")

    return output_path


def main():
    parser = argparse.ArgumentParser(description='Convert YOLO model to ONNX format')
    parser.add_argument('model', type=str, help='Path to .pt model file')
    parser.add_argument('--imgsz', type=int, default=640, help='Input image size (default: 640)')
    parser.add_argument('--no-simplify', action='store_true', help='Do not simplify ONNX model')

    args = parser.parse_args()

    convert_to_onnx(
        model_path=args.model,
        imgsz=args.imgsz,
        simplify=not args.no_simplify
    )


if __name__ == '__main__':
    main()
