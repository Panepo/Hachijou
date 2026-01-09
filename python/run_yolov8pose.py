# run_yolov8pose.py
import argparse
import time
import cv2
import numpy as np
import os
from datetime import datetime
from yolov8pose import YOLOv8PoseONNX


def parse_args():
    parser = argparse.ArgumentParser(description="Run YOLOv8-Pose ONNX on image or webcam.")

    # Model settings
    parser.add_argument("--model", type=str, default="./models/yolov8m-pose.onnx",
                       help="Path to YOLOv8-Pose ONNX model.")
    parser.add_argument("--size", type=int, default=640,
                       help="Model input size (e.g., 640 for 640x640).")
    parser.add_argument("--conf", type=float, default=0.5,
                       help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.4,
                       help="IoU threshold for NMS.")

    # Visualization settings
    parser.add_argument("--kpt-threshold", type=float, default=0.5,
                       help="Keypoint confidence threshold for visualization.")
    parser.add_argument("--no-bbox", action="store_true",
                       help="Don't draw bounding boxes.")
    parser.add_argument("--no-keypoints", action="store_true",
                       help="Don't draw keypoints.")
    parser.add_argument("--no-skeleton", action="store_true",
                       help="Don't draw skeleton.")

    # Input settings
    parser.add_argument("--image", type=str,
                       help="Path to input image (single image mode).")
    parser.add_argument("--output", type=str, default="pose_output.jpg",
                       help="Path to save output image.")
    parser.add_argument("--webcam", action="store_true",
                       help="Use webcam as input.")
    parser.add_argument("--cam-id", type=int, default=0,
                       help="Webcam device ID.")
    parser.add_argument("--flip", action="store_true",
                       help="Flip webcam horizontally.")
    parser.add_argument("--video", type=str,
                       help="Path to input video file.")

    # Output settings
    parser.add_argument("--save-video", type=str,
                       help="Path to save output video.")
    parser.add_argument("--screenshot-dir", type=str, default="./screenshots",
                       help="Directory to save screenshots.")
    parser.add_argument("--no-show", action="store_true",
                       help="Don't display window.")

    return parser.parse_args()


def run_image(args):
    """Run pose detection on a single image"""
    # Initialize detector
    detector = YOLOv8PoseONNX(
        model_path=args.model,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        input_size=(args.size, args.size)
    )

    # Load image
    image = cv2.imread(args.image)
    if image is None:
        raise RuntimeError(f"Failed to load image: {args.image}")

    print(f"\nRunning pose detection on: {args.image}")
    print(f"Image size: {image.shape[1]}x{image.shape[0]}")

    # Detect poses
    t0 = time.time()
    detections, result = detector.detect_and_draw(
        image,
        draw_bbox=not args.no_bbox,
        draw_keypoints=not args.no_keypoints,
        draw_skeleton=not args.no_skeleton,
        kpt_threshold=args.kpt_threshold
    )
    infer_time = (time.time() - t0) * 1000.0

    # Save result
    os.makedirs("./output", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"./output/{timestamp}.jpg"
    cv2.imwrite(output_path, result)

    # Print results
    print(f"\n{'='*60}")
    print(f"Inference time: {infer_time:.1f}ms")
    print(f"Found {len(detections)} person(s)")
    print(f"Output saved to: {args.output}")
    print(f"{'='*60}\n")

    for i, det in enumerate(detections):
        print(f"Person {i+1}:")
        print(f"  Confidence: {det['confidence']:.3f}")
        bbox = det['bbox']
        print(f"  Bounding box: [{bbox[0]:.0f}, {bbox[1]:.0f}, {bbox[2]:.0f}, {bbox[3]:.0f}]")

        # Count visible keypoints
        visible_kpts = sum(1 for kpt in det['keypoints'] if kpt[2] > args.kpt_threshold)
        print(f"  Visible keypoints: {visible_kpts}/17")

        # Show keypoint details
        print(f"  Keypoints:")
        for j, (x, y, conf) in enumerate(det['keypoints']):
            if conf > args.kpt_threshold:
                print(f"    {detector.keypoint_names[j]:15s}: ({x:6.1f}, {y:6.1f}) [conf: {conf:.3f}]")
        print()

    # Display if not suppressed
    if not args.no_show:
        cv2.imshow("YOLOv8-Pose Detection", result)
        print("Press any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def save_screenshot(frame, screenshot_dir):
    """Save a screenshot with timestamp"""
    os.makedirs(screenshot_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"pose_screenshot_{timestamp}.jpg"
    filepath = os.path.join(screenshot_dir, filename)
    cv2.imwrite(filepath, frame)
    return filepath


def run_video_stream(args, cap, source_name):
    """Run pose detection on video stream (webcam or video file)"""
    # Initialize detector
    detector = YOLOv8PoseONNX(
        model_path=args.model,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        input_size=(args.size, args.size)
    )

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_read = cap.get(cv2.CAP_PROP_FPS) or 30.0

    print(f"\n{'='*60}")
    print(f"Source: {source_name}")
    print(f"Resolution: {width}x{height}")
    print(f"FPS: {fps_read:.1f}")
    print(f"Model: {args.model}")
    print(f"Input size: {args.size}x{args.size}")
    print(f"Confidence threshold: {args.conf}")
    print(f"IoU threshold: {args.iou}")
    print(f"Keypoint threshold: {args.kpt_threshold}")
    print(f"{'='*60}")
    print("\nControls:")
    print("  's' - Save screenshot")
    print("  'b' - Toggle bounding boxes")
    print("  'k' - Toggle keypoints")
    print("  'l' - Toggle skeleton")
    print("  'q' or ESC - Exit")
    print()

    # Video writer
    writer = None
    if args.save_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(args.save_video, fourcc, fps_read, (width, height))
        if writer.isOpened():
            print(f"Saving video to: {args.save_video}")
        else:
            print(f"Warning: Failed to open video writer")
            writer = None

    # FPS tracking
    frame_count = 0
    start_time = time.time()
    avg_infer_ms = 0.0
    alpha = 0.1  # EMA smoothing factor

    # Toggle states
    show_bbox = not args.no_bbox
    show_keypoints = not args.no_keypoints
    show_skeleton = not args.no_skeleton

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("\nEnd of stream.")
                break

            if args.flip:
                frame = cv2.flip(frame, 1)

            # Detect poses
            t0 = time.time()
            detections, vis = detector.detect_and_draw(
                frame,
                draw_bbox=show_bbox,
                draw_keypoints=show_keypoints,
                draw_skeleton=show_skeleton,
                kpt_threshold=args.kpt_threshold
            )
            infer_ms = (time.time() - t0) * 1000.0

            # Update average inference time
            if avg_infer_ms == 0.0:
                avg_infer_ms = infer_ms
            else:
                avg_infer_ms = alpha * infer_ms + (1 - alpha) * avg_infer_ms

            # Calculate FPS
            frame_count += 1
            elapsed_time = time.time() - start_time
            fps_avg = frame_count / elapsed_time if elapsed_time > 0 else 0

            # Count total visible keypoints
            total_keypoints = 0
            for det in detections:
                total_keypoints += sum(1 for kpt in det['keypoints'] if kpt[2] > args.kpt_threshold)

            # Draw HUD
            hud_y = 30
            cv2.putText(vis, f"FPS: {fps_avg:.1f}", (10, hud_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            hud_y += 30
            cv2.putText(vis, f"Infer: {infer_ms:.1f}ms (avg: {avg_infer_ms:.1f}ms)", (10, hud_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            hud_y += 30
            cv2.putText(vis, f"Persons: {len(detections)} | Keypoints: {total_keypoints}", (10, hud_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Draw toggle states
            hud_y += 30
            toggles = f"Bbox: {'ON' if show_bbox else 'OFF'} | Kpts: {'ON' if show_keypoints else 'OFF'} | Skel: {'ON' if show_skeleton else 'OFF'}"
            cv2.putText(vis, toggles, (10, hud_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # Write video frame
            if writer is not None:
                writer.write(vis)

            # Show window
            if not args.no_show:
                cv2.imshow("YOLOv8-Pose", vis)

                key = cv2.waitKey(1) & 0xFF

                if key == ord('q') or key == 27:  # q or ESC
                    print("\nExiting...")
                    break
                elif key == ord('s'):  # Save screenshot
                    filepath = save_screenshot(vis, args.screenshot_dir)
                    print(f"Screenshot saved: {filepath}")
                elif key == ord('b'):  # Toggle bbox
                    show_bbox = not show_bbox
                    print(f"Bounding boxes: {'ON' if show_bbox else 'OFF'}")
                elif key == ord('k'):  # Toggle keypoints
                    show_keypoints = not show_keypoints
                    print(f"Keypoints: {'ON' if show_keypoints else 'OFF'}")
                elif key == ord('l'):  # Toggle skeleton
                    show_skeleton = not show_skeleton
                    print(f"Skeleton: {'ON' if show_skeleton else 'OFF'}")

    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

        # Print statistics
        print(f"\n{'='*60}")
        print(f"Session Statistics:")
        print(f"  Total frames: {frame_count}")
        print(f"  Average FPS: {fps_avg:.2f}")
        print(f"  Average inference time: {avg_infer_ms:.2f}ms")
        print(f"{'='*60}\n")


def run_webcam(args):
    """Run pose detection on webcam"""
    cap = cv2.VideoCapture(args.cam_id)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open webcam with ID {args.cam_id}")

    run_video_stream(args, cap, f"Webcam {args.cam_id}")


def run_video(args):
    """Run pose detection on video file"""
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video file: {args.video}")

    run_video_stream(args, cap, args.video)


def main():
    args = parse_args()

    try:
        if args.image:
            run_image(args)
        elif args.webcam:
            run_webcam(args)
        elif args.video:
            run_video(args)
        else:
            print("Error: Please specify input source (--image, --webcam, or --video)")
            print("Use --help for more information")
            return 1

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
