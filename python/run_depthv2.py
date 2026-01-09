# run_depthv2.py
import argparse
import time
import cv2
from datetime import datetime
from depthv2 import DepthAnythingV2


def parse_args():
    parser = argparse.ArgumentParser(description="Run Depth Anything V2 ONNX on image or webcam.")

    # Model settings
    parser.add_argument("--model", type=str, default="./models/depth_anything_v2_vits.onnx",
                       help="Path to Depth Anything V2 ONNX model.")
    parser.add_argument("--height", type=int, default=518, help="Model input height.")
    parser.add_argument("--width", type=int, default=518, help="Model input width.")
    parser.add_argument("--colormap", type=str, default="inferno",
                       choices=["inferno", "plasma", "viridis", "jet", "turbo", "hot", "cool", "gray"],
                       help="Colormap for depth visualization.")

    # Single image mode
    parser.add_argument("--image", type=str, help="Path to input image.")
    parser.add_argument("--output", type=str, default="depth_output.jpg",
                       help="Path to save output image.")

    # Webcam / video mode
    parser.add_argument("--webcam", action="store_true", help="Use webcam as input.")
    parser.add_argument("--cam-id", type=int, default=0, help="Webcam device ID.")
    parser.add_argument("--save-video", type=str, help="Path to save output video.")
    parser.add_argument("--screenshot-dir", type=str, default="./screenshots",
                       help="Directory to save screenshots (default: ./screenshots).")
    parser.add_argument("--no-show", action="store_true", help="Don't display video window.")
    parser.add_argument("--flip", action="store_true", help="Flip webcam horizontally.")
    parser.add_argument("--side-by-side", action="store_true",
                       help="Show original and depth side-by-side.")

    return parser.parse_args()


def get_colormap(name):
    """Convert colormap name to OpenCV colormap constant"""
    colormaps = {
        "inferno": cv2.COLORMAP_INFERNO,
        "plasma": cv2.COLORMAP_PLASMA,
        "viridis": cv2.COLORMAP_VIRIDIS,
        "jet": cv2.COLORMAP_JET,
        "turbo": cv2.COLORMAP_TURBO,
        "hot": cv2.COLORMAP_HOT,
        "cool": cv2.COLORMAP_COOL,
        "gray": cv2.COLORMAP_BONE,
    }
    return colormaps.get(name.lower(), cv2.COLORMAP_INFERNO)


def save_screenshot(image, screenshot_dir="./screenshots"):
    """
    Save a screenshot with timestamp

    Args:
        image: Image to save
        screenshot_dir: Directory to save screenshots

    Returns:
        str: Path to saved screenshot
    """
    import os

    # Create directory if it doesn't exist
    os.makedirs(screenshot_dir, exist_ok=True)

    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"screenshot_{timestamp}.jpg"
    filepath = os.path.join(screenshot_dir, filename)

    # Save image
    cv2.imwrite(filepath, image)

    return filepath


def run_image(args):
    """Run Depth Anything V2 on a single image"""
    colormap = get_colormap(args.colormap)

    # Initialize depth estimator
    depth_estimator = DepthAnythingV2(
        model_path=args.model,
        input_size=(args.height, args.width)
    )
    depth_estimator.colormap = colormap

    # Load image
    image = cv2.imread(args.image)
    if image is None:
        raise RuntimeError(f"Failed to load image: {args.image}")

    print(f"Processing {args.image}...")
    t0 = time.time()

    # Predict depth
    depth_map, colored_depth = depth_estimator.predict_and_visualize(image, colormap)

    inference_time = (time.time() - t0) * 1000.0

    # Create output
    if args.side_by_side:
        output = depth_estimator.create_side_by_side(image, colored_depth)
    else:
        output = colored_depth

    # Save result
    import os
    os.makedirs("./output", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"./output/{timestamp}.jpg"
    cv2.imwrite(output_path, output)

    print(f"Saved depth map to {output_path}")
    print(f"Inference time: {inference_time:.1f}ms")
    print(f"Depth range: {depth_map.min():.2f} to {depth_map.max():.2f}")

    # Display result
    cv2.imshow("Depth Estimation", output)
    print("Press any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def run_webcam(args):
    """Run Depth Anything V2 on webcam stream"""
    colormap = get_colormap(args.colormap)

    # Initialize depth estimator
    depth_estimator = DepthAnythingV2(
        model_path=args.model,
        input_size=(args.height, args.width)
    )
    depth_estimator.colormap = colormap

    # Open webcam
    cap = cv2.VideoCapture(args.cam_id)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open webcam with id {args.cam_id}")

    # Get camera properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_read = cap.get(cv2.CAP_PROP_FPS) or 30.0

    print(f"Webcam opened: {width}x{height} @ ~{fps_read:.1f} FPS")
    print(f"Model: {args.model}")
    print(f"Input size: {args.height}x{args.width}")
    print(f"Colormap: {args.colormap}")
    print("Press 's' to save screenshot, 'c' to change colormap, 'q' or ESC to exit.")

    # Video writer if saving
    writer = None
    if args.save_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_width = width * 2 if args.side_by_side else width
        writer = cv2.VideoWriter(args.save_video, fourcc, fps_read, (output_width, height))
        if not writer.isOpened():
            print(f"Warning: failed to open video writer at {args.save_video}")
            writer = None
        else:
            print(f"Saving video to {args.save_video}")

    # FPS tracking
    avg_infer_ms = 0.0
    alpha = 0.1  # smoothing factor for EMA
    frame_count = 0
    start_time = time.time()

    window_name = "Depth Anything V2 - Webcam"

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("Frame grab failed; stopping.")
                break

            if args.flip:
                frame = cv2.flip(frame, 1)

            # Inference
            t0 = time.time()
            depth_map, colored_depth = depth_estimator.predict_and_visualize(frame, colormap)
            infer_ms = (time.time() - t0) * 1000.0
            avg_infer_ms = alpha * infer_ms + (1 - alpha) * avg_infer_ms if avg_infer_ms > 0 else infer_ms

            # Calculate FPS
            frame_count += 1
            elapsed_time = time.time() - start_time
            fps_avg = frame_count / elapsed_time if elapsed_time > 0 else 0

            # Create visualization
            if args.side_by_side:
                vis = depth_estimator.create_side_by_side(frame, colored_depth)
            else:
                vis = colored_depth

            # Draw HUD (FPS and latency info)
            hud = f"FPS: {fps_avg:.1f}  Infer: {infer_ms:.1f}ms  Avg: {avg_infer_ms:.1f}ms"
            cv2.putText(vis, hud, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

            # Add depth range info
            depth_info = f"Depth: {depth_map.min():.2f} - {depth_map.max():.2f}"
            cv2.putText(vis, depth_info, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

            # Show window
            if not args.no_show:
                cv2.imshow(window_name, vis)
                # Exit with q or ESC
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    print("Exit requested.")
                    break
                # Save screenshot with 's' key
                elif key == ord('s'):
                    screenshot_path = save_screenshot(vis, args.screenshot_dir)
                    print(f"\n📸 Screenshot saved: {screenshot_path}")
                # Change colormap with 'c' key
                elif key == ord('c'):
                    colormaps_list = [
                        cv2.COLORMAP_INFERNO, cv2.COLORMAP_PLASMA, cv2.COLORMAP_VIRIDIS,
                        cv2.COLORMAP_JET, cv2.COLORMAP_TURBO, cv2.COLORMAP_HOT
                    ]
                    current_idx = colormaps_list.index(colormap) if colormap in colormaps_list else 0
                    colormap = colormaps_list[(current_idx + 1) % len(colormaps_list)]
                    depth_estimator.colormap = colormap
                    print(f"Changed colormap to index {colormaps_list.index(colormap)}")

            # Save video frame
            if writer is not None:
                writer.write(vis)

    finally:
        # Cleanup
        cap.release()
        if writer is not None:
            writer.release()
            print(f"Video saved to {args.save_video}")
        if not args.no_show:
            cv2.destroyAllWindows()

        # Print statistics
        total_time = time.time() - start_time
        print(f"\nSession statistics:")
        print(f"  Total frames: {frame_count}")
        print(f"  Total time: {total_time:.1f}s")
        print(f"  Average FPS: {frame_count / total_time:.1f}")
        print(f"  Average inference time: {avg_infer_ms:.1f}ms")


def main():
    args = parse_args()

    # Priority: if --image is provided, run single-image mode
    if args.image:
        run_image(args)
        return

    # Otherwise, webcam mode if requested
    if args.webcam:
        run_webcam(args)
        return

    # If neither provided, show usage hint
    print("Nothing to run. Provide --image for single image, or --webcam for live camera.")
    print("\nExamples:")
    print("  Image:   python run_depthv2.py --image test.jpg --output depth_result.jpg --side-by-side")
    print("  Webcam:  python run_depthv2.py --webcam --side-by-side")
    print("  Save:    python run_depthv2.py --webcam --save-video depth_output.mp4 --colormap plasma")


if __name__ == "__main__":
    main()
