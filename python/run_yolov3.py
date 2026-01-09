
# run_yolov3.py
import argparse
import time
import cv2
import numpy as np
import os
from datetime import datetime
from yolov3 import YOLOv3ONNX

def parse_args():
    parser = argparse.ArgumentParser(description="Run YOLOv3 ONNX on image or webcam/video.")
    # Model / thresholds
    parser.add_argument("--model", type=str, default="./models/yolov3-12.onnx", help="Path to YOLOv3 ONNX model.")
    parser.add_argument("--height", type=int, default=416, help="Model input height (e.g., 416).")
    parser.add_argument("--width", type=int, default=416, help="Model input width (e.g., 416).")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold.")
    parser.add_argument("--nms", type=float, default=0.4, help="NMS threshold.")
    parser.add_argument("--gpu", action="store_true", help="Use GPU inference (CUDA).")

    # Single image mode
    parser.add_argument("--image", type=str, help="Path to input image (if provided, runs single-image inference).")
    parser.add_argument("--output", type=str, default="output.jpg", help="Path to save output image.")

    # Webcam / video mode
    parser.add_argument("--webcam", action="store_true", help="Use webcam as input.")
    parser.add_argument("--cam-id", type=int, default=0, help="Webcam device ID (0 is default camera).")
    parser.add_argument("--save-video", type=str, help="Path to save output video.")
    parser.add_argument("--screenshot-dir", type=str, default="./screenshots",
                       help="Directory to save screenshots (default: ./screenshots).")
    parser.add_argument("--no-show", action="store_true", help="Don't display video window.")
    parser.add_argument("--flip", action="store_true", help="Flip webcam horizontally.")

    return parser.parse_args()

def run_image(args):
    """Run YOLOv3 detection on a single image"""
    # Initialize detector
    yolo = YOLOv3ONNX(
        model_path=args.model,
        conf_threshold=args.conf,
        nms_threshold=args.nms,
        input_size=(args.height, args.width)
    )

    # Load image
    image = cv2.imread(args.image)
    if image is None:
        raise RuntimeError(f"Failed to load image: {args.image}")

    # Detect and draw
    print(f"Running detection on {args.image}...")
    t0 = time.time()
    detections, result = yolo.detect_and_draw(image)
    infer_time = (time.time() - t0) * 1000.0

    # Save result
    os.makedirs("./output", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"./output/{timestamp}.jpg"
    cv2.imwrite(output_path, result)
    print(f"Saved detections to {output_path}")
    print(f"Inference time: {infer_time:.1f}ms")
    print(f"Found {len(detections)} objects:")

    for i, det in enumerate(detections):
        class_id, conf, x1, y1, x2, y2 = det
        class_name = yolo.class_names[int(class_id)]
        print(f"  {i+1}. {class_name}: {conf:.3f} at [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")

def save_screenshot(frame, screenshot_dir):
    """Save a screenshot with timestamp to the specified directory"""
    os.makedirs(screenshot_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"screenshot_{timestamp}.jpg"
    filepath = os.path.join(screenshot_dir, filename)
    cv2.imwrite(filepath, frame)
    return filepath

def run_webcam(args):
    """Run YOLOv3 detection on webcam stream"""
    # Initialize detector
    yolo = YOLOv3ONNX(
        model_path=args.model,
        conf_threshold=args.conf,
        nms_threshold=args.nms,
        input_size=(args.height, args.width)
    )

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
    print(f"Confidence threshold: {args.conf}")
    print(f"NMS threshold: {args.nms}")
    print("Press 's' to save screenshot, 'q' or ESC to exit.")

    # Video writer if saving
    writer = None
    if args.save_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(args.save_video, fourcc, fps_read, (width, height))
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
            detections, vis = yolo.detect_and_draw(frame)
            infer_ms = (time.time() - t0) * 1000.0
            avg_infer_ms = alpha * infer_ms + (1 - alpha) * avg_infer_ms if avg_infer_ms > 0 else infer_ms

            # Calculate FPS
            frame_count += 1
            elapsed_time = time.time() - start_time
            fps_avg = frame_count / elapsed_time if elapsed_time > 0 else 0

            # Draw HUD (FPS and latency info)
            hud = f"FPS: {fps_avg:.1f}  Infer: {infer_ms:.1f}ms  Avg: {avg_infer_ms:.1f}ms  Objects: {len(detections)}"
            cv2.putText(vis, hud, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

            # Show window
            if not args.no_show:
                cv2.imshow("YOLOv3 ONNX", vis)
                # Exit with q or ESC
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    print("Exit requested.")
                    break
                # Save screenshot with 's' key
                elif key == ord('s'):
                    screenshot_path = save_screenshot(vis, args.screenshot_dir)
                    print(f"\n📸 Screenshot saved: {screenshot_path}")

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

    # Priority: if --image is provided, run single-image mode.
    if args.image:
        run_image(args)
        return

    # Otherwise, webcam mode if requested.
    if args.webcam:
        run_webcam(args)
        return

    # If neither provided, show usage hint.
    print("Nothing to run. Provide --image for single image, or --webcam for live camera.")
    print("\nExamples:")
    print("  Image:   python run_yolov3.py --image test.jpg --output result.jpg")
    print("  Webcam:  python run_yolov3.py --webcam --gpu")
    print("  Save:    python run_yolov3.py --webcam --save-video output.mp4")

if __name__ == "__main__":
    main()

