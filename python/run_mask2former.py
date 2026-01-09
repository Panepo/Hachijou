
# run_mask2former.py
import argparse
import time
import cv2
import numpy as np
import os
from datetime import datetime
from mask2former import Mask2FormerONNX

def parse_args():
    parser = argparse.ArgumentParser(description="Run Mask2Former ONNX on image or webcam/video.")
    # Model / thresholds
    parser.add_argument("--model", type=str, default="./models/Mask2Former.onnx",
                       help="Path to Mask2Former ONNX model.")
    parser.add_argument("--height", type=int, default=384, help="Model input height (e.g., 384).")
    parser.add_argument("--width", type=int, default=384, help="Model input width (e.g., 384).")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold.")
    parser.add_argument("--alpha", type=float, default=0.6, help="Overlay transparency (0-1).")
    parser.add_argument("--no-legend", action="store_true", help="Don't show class legend.")

    # Single image mode
    parser.add_argument("--image", type=str, help="Path to input image (if provided, runs single-image inference).")
    parser.add_argument("--output", type=str, default="output.jpg", help="Path to save output image.")

    # Webcam / video mode
    parser.add_argument("--webcam", action="store_true", help="Use webcam as input.")
    parser.add_argument("--video", type=str, help="Path to input video file.")
    parser.add_argument("--cam-id", type=int, default=0, help="Webcam device ID (0 is default camera).")
    parser.add_argument("--save-video", type=str, help="Path to save output video.")
    parser.add_argument("--screenshot-dir", type=str, default="./screenshots",
                       help="Directory to save screenshots (default: ./screenshots).")
    parser.add_argument("--no-show", action="store_true", help="Don't display video window.")
    parser.add_argument("--flip", action="store_true", help="Flip webcam horizontally.")

    return parser.parse_args()

def run_image(args):
    """Run Mask2Former segmentation on a single image"""
    # Initialize segmentation model
    mask2former = Mask2FormerONNX(
        model_path=args.model,
        conf_threshold=args.conf,
        input_size=(args.height, args.width)
    )

    # Load image
    image = cv2.imread(args.image)
    if image is None:
        raise RuntimeError(f"Failed to load image: {args.image}")

    # Segment and draw
    print(f"Running segmentation on {args.image}...")
    t0 = time.time()
    seg_map, colored_mask, result = mask2former.detect_and_draw(image, alpha=args.alpha,
                                                                 show_legend=not args.no_legend)
    infer_time = (time.time() - t0) * 1000.0

    # Save result
    os.makedirs("./output", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"./output/mask2former_{timestamp}.jpg"
    cv2.imwrite(output_path, result)
    print(f"Saved segmentation to {output_path}")
    print(f"Inference time: {infer_time:.1f}ms")

    # Get and print class statistics
    stats = mask2former.get_class_statistics(seg_map)
    print(f"\nDetected classes ({len(stats)}):")
    total_pixels = sum(stats.values())
    for class_name, count in stats.items():
        percentage = (count / total_pixels) * 100
        print(f"  {class_name}: {count:,} pixels ({percentage:.1f}%)")

def save_screenshot(frame, screenshot_dir):
    """Save a screenshot with timestamp to the specified directory"""
    os.makedirs(screenshot_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"screenshot_{timestamp}.jpg"
    filepath = os.path.join(screenshot_dir, filename)
    cv2.imwrite(filepath, frame)
    return filepath

def run_video(args):
    """Run Mask2Former segmentation on video or webcam stream"""
    # Initialize segmentation model
    mask2former = Mask2FormerONNX(
        model_path=args.model,
        conf_threshold=args.conf,
        input_size=(args.height, args.width)
    )

    # Open video source
    if args.webcam:
        cap = cv2.VideoCapture(args.cam_id)
        source_name = f"Webcam {args.cam_id}"
    elif args.video:
        cap = cv2.VideoCapture(args.video)
        source_name = args.video
    else:
        raise RuntimeError("No video source specified. Use --webcam or --video")

    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video source: {source_name}")

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_read = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video source: {source_name}")
    print(f"Resolution: {width}x{height} @ ~{fps_read:.1f} FPS")
    if args.video:
        print(f"Total frames: {total_frames}")
    print(f"Model: {args.model}")
    print(f"Input size: {args.height}x{args.width}")
    print(f"Confidence threshold: {args.conf}")
    print(f"Alpha (transparency): {args.alpha}")
    print(f"Show legend: {not args.no_legend}")
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
    alpha_ema = 0.1  # smoothing factor for EMA
    frame_count = 0
    start_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("Frame grab failed or end of video; stopping.")
                break

            if args.flip:
                frame = cv2.flip(frame, 1)

            # Inference
            t0 = time.time()
            seg_map, colored_mask, vis = mask2former.detect_and_draw(frame, alpha=args.alpha,
                                                                      show_legend=not args.no_legend)
            infer_ms = (time.time() - t0) * 1000.0
            avg_infer_ms = alpha_ema * infer_ms + (1 - alpha_ema) * avg_infer_ms if avg_infer_ms > 0 else infer_ms

            # Calculate FPS
            frame_count += 1
            elapsed_time = time.time() - start_time
            fps_avg = frame_count / elapsed_time if elapsed_time > 0 else 0

            # Get unique classes
            unique_classes = len(np.unique(seg_map))

            # Draw HUD (FPS and latency info) - place at bottom to avoid legend
            hud_y = vis.shape[0] - 10
            hud = f"FPS: {fps_avg:.1f}  Infer: {infer_ms:.1f}ms  Avg: {avg_infer_ms:.1f}ms  Classes: {unique_classes}"

            # Draw black background for text
            (text_width, text_height), _ = cv2.getTextSize(hud, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(vis, (5, hud_y - text_height - 5), (15 + text_width, hud_y + 5), (0, 0, 0), -1)
            cv2.putText(vis, hud, (10, hud_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

            # Show progress for video files
            if args.video:
                progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
                progress_text = f"Progress: {frame_count}/{total_frames} ({progress:.1f}%)"
                cv2.putText(vis, progress_text, (10, vis.shape[0] - 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

            # Show window
            if not args.no_show:
                cv2.imshow("Mask2Former Segmentation", vis)
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

    # Video or webcam mode
    if args.webcam or args.video:
        run_video(args)
        return

    # If nothing provided, show usage hint.
    print("Nothing to run. Provide --image for single image, --webcam for live camera, or --video for video file.")
    print("\nExamples:")
    print("  Image:   python run_mask2former.py --image test.jpg")
    print("  Webcam:  python run_mask2former.py --webcam")
    print("  Video:   python run_mask2former.py --video input.mp4 --save-video output.mp4")
    print("  Custom:  python run_mask2former.py --webcam --alpha 0.7 --no-legend")

if __name__ == "__main__":
    main()
