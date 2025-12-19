import onnx
import argparse
import os
from pathlib import Path


def merge_onnx_model(input_path, output_path=None):
    """
    Merge ONNX model with external data into a single file.

    Args:
        input_path: Path to the .onnx file (model.data should be in same directory)
        output_path: Path for the merged output file (optional)
    """
    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Model file not found: {input_path}")

    # Default output path
    if output_path is None:
        output_path = input_path.parent / f"{input_path.stem}_merged.onnx"
    else:
        output_path = Path(output_path)

    print(f"Loading model from: {input_path}")

    # Load model (automatically loads external data if present)
    model = onnx.load(str(input_path))

    print(f"Model loaded successfully")
    print(f"Saving merged model to: {output_path}")

    # Save as single file with all data embedded
    onnx.save(model, str(output_path), save_as_external_data=False)

    # Get file sizes
    input_size = input_path.stat().st_size / (1024 * 1024)  # MB
    output_size = output_path.stat().st_size / (1024 * 1024)  # MB

    print(f"\n✓ Merge complete!")
    print(f"  Input size:  {input_size:.2f} MB")
    print(f"  Output size: {output_size:.2f} MB")
    print(f"  Saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Merge ONNX model with external data into a single file"
    )
    parser.add_argument(
        "input",
        type=str,
        help="Path to input .onnx file (model.data should be in same directory)"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Path to output merged .onnx file (default: input_merged.onnx)"
    )

    args = parser.parse_args()

    try:
        merge_onnx_model(args.input, args.output)
    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
