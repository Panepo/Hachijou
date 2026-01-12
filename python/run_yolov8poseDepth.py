# run_yolov8poseDepth.py
import argparse
import time
import cv2
import os
from datetime import datetime
from yolov8poseDepth import YOLOv8PoseDepth


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run YOLOv8-Pose for person distance estimation with occlusion detection."
    )

    # Model settings
    parser.add_argument("--pose-model", type=str, default="./models/yolov8m-pose.onnx",
                       help="Path to YOLOv8-Pose ONNX model.")
    parser.add_argument("--pose-size", type=int, default=640,
                       help="YOLOv8-Pose input size.")
    parser.add_argument("--conf", type=float, default=0.5,
                       help="Pose detection confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.4,
                       help="IoU threshold for NMS.")
    parser.add_argument("--kpt-threshold", type=float, default=0.5,
                       help="Keypoint confidence threshold.")

    # Distance estimation settings
    parser.add_argument("--person-height", type=float, default=180.0,
                       help="Assumed person height in cm (default: 180.0cm).")
    parser.add_argument("--focal-length", type=float, default=None,
                       help="Camera focal length in pixels (auto-calibrate if not provided).")

    # Occlusion detection settings
    parser.add_argument("--min-keypoints", type=int, default=10,
                       help="Minimum visible keypoints to consider person not occluded (default: 10/17).")

    # Border detection settings
    parser.add_argument("--border-threshold", type=int, default=10,
                       help="Pixel threshold for detecting if bbox is on border (default: 10px).")
    parser.add_argument("--border-distance-factor", type=float, default=0.6,
                       help="Multiply estimated distance by this factor if on border/very close (default: 0.6).")

    # Input settings
    parser.add_argument("--image", type=str, help="Path to input image.")
    parser.add_argument("--output", type=str, default="pose_depth_output.jpg",
                       help="Path to save output image.")
    parser.add_argument("--webcam", action="store_true", help="Use webcam as input.")
    parser.add_argument("--cam-id", type=int, default=0, help="Webcam device ID.")
    parser.add_argument("--flip", action="store_true", help="Flip webcam horizontally.")
    parser.add_argument("--video", type=str, help="Path to input video file.")

    # Visualization settings
    parser.add_argument("--save-video", type=str, help="Path to save output video.")
    parser.add_argument("--no-show", action="store_true", help="Don't display video window.")
    parser.add_argument("--screenshot-dir", type=str, default="./screenshots",
                       help="Directory to save screenshots.")
    parser.add_argument("--no-keypoints", action="store_true",
                       help="Don't draw keypoints.")
    parser.add_argument("--no-skeleton", action="store_true",
                       help="Don't draw skeleton.")
    parser.add_argument("--calibrate", action="store_true",
                       help="Calibrate focal length (stand at known distance and press 'c').")
    parser.add_argument("--calibrate-distance", type=float, default=200.0,
                       help="Known distance for calibration in cm (default: 200.0cm).")

    return parser.parse_args()


def save_screenshot(image, screenshot_dir="./screenshots"):
    """Save a screenshot with timestamp"""
    os.makedirs(screenshot_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"pose_depth_screenshot_{timestamp}.jpg"
    filepath = os.path.join(screenshot_dir, filename)
    cv2.imwrite(filepath, image)
    return filepath


def run_image(args, depth_estimator):
    """Process single image"""
    image = cv2.imread(args.image)
    if image is None:
        raise RuntimeError(f"Failed to load image: {args.image}")

    # Auto-estimate focal length if not provided
    if depth_estimator.focal_length is None:
        h, w = image.shape[:2]
        depth_estimator.focal_length = depth_estimator.estimate_focal_length(w, h)
        print(f"Auto-estimated focal length: {depth_estimator.focal_length:.1f}px")

    print(f"\nProcessing {args.image}...")

    # Process frame
    t0 = time.time()
    output, person_count, person_detections = depth_estimator.process_frame(
        image, not args.no_keypoints, not args.no_skeleton
    )
    pose_time = (time.time() - t0) * 1000.0

    # Print detection details
    print(f"\n{'='*70}")
    print(f"Detected {person_count} person(s)")
    print(f"Pose detection: {pose_time:.1f}ms")
    print(f"{'='*70}\n")

    for i, person in enumerate(person_detections, 1):
        print(f"Person {i}:")
        print(f"  Confidence: {person['confidence']:.3f}")
        print(f"  Visible keypoints: {person['occlusion']['visible_keypoints']}/17")

        if person['posture']['confidence'] > 0.3:
            posture_status = "Standing" if person['posture']['is_standing'] else "Sitting/Lying"
            print(f"  Posture: {posture_status} (confidence: {person['posture']['confidence']:.1%})")

        if person['distance_cm'] is not None:
            dist_str = f"{person['distance_cm']/100:.2f}m" if person['distance_cm'] >= 100 else f"{person['distance_cm']:.0f}cm"
            print(f"  Distance: {dist_str}")

        print(f"  Bbox height: {person['bbox_height']:.1f}px")
        print()

    # Save result
    os.makedirs("./output", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"./output/{timestamp}.jpg"
    cv2.imwrite(output_path, output)
    print(f"Saved result to: {output_path}\n")

    # Display
    if not args.no_show:
        cv2.imshow("YOLOv8-Pose Distance Estimation", output)
        print("Press any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def run_video_stream(args, cap, source_name, depth_estimator):
    """Run on video stream (webcam or video file)"""
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_read = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # Auto-estimate focal length if not provided
    if depth_estimator.focal_length is None:
        depth_estimator.focal_length = depth_estimator.estimate_focal_length(width, height)
        print(f"Auto-estimated focal length: {depth_estimator.focal_length:.1f}px")

    print(f"\n{'='*70}")
    print(f"Source: {source_name}")
    print(f"Resolution: {width}x{height} @ {fps_read:.1f} FPS")
    print(f"Person height: {depth_estimator.person_height_cm}cm")
    print(f"Focal length: {depth_estimator.focal_length:.1f}px")
    if args.calibrate:
        print(f"Calibration: Press 'c' when at {args.calibrate_distance}cm")
    print(f"{'='*70}")
    print("\nControls:")
    print("  's' - Save screenshot")
    print("  'c' - Calibrate focal length")
    print("  'k' - Toggle keypoints")
    print("  'q' or ESC - Exit\n")

    # Video writer
    writer = None
    if args.save_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(args.save_video, fourcc, fps_read, (width, height))
        if writer.isOpened():
            print(f"Saving video to: {args.save_video}")
        else:
            print("Warning: Failed to open video writer")
            writer = None

    # State variables
    frame_count = 0
    start_time = time.time()
    avg_pose_ms = 0.0
    alpha = 0.1
    draw_keypoints = not args.no_keypoints
    draw_skeleton = not args.no_skeleton

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("\nEnd of stream.")
                break

            if args.flip:
                frame = cv2.flip(frame, 1)

            # Process frame
            t0 = time.time()
            output, person_count, person_detections = depth_estimator.process_frame(
                frame, draw_keypoints, draw_skeleton
            )
            total_time = (time.time() - t0) * 1000.0

            # Update averages
            avg_pose_ms = alpha * total_time + (1 - alpha) * avg_pose_ms if avg_pose_ms > 0 else total_time

            # Calculate FPS
            frame_count += 1
            elapsed = time.time() - start_time
            fps_avg = frame_count / elapsed if elapsed > 0 else 0

            # Count status
            too_close_count = sum(1 for p in person_detections if p['border'].get('too_close', False))
            border_count = sum(1 for p in person_detections if p['border']['on_border'])
            standing_count = sum(1 for p in person_detections
                               if p['posture']['is_standing'] and p['posture']['confidence'] > 0.5)
            sitting_count = sum(1 for p in person_detections
                              if not p['posture']['is_standing'] and p['posture']['confidence'] > 0.5)

            # Draw HUD
            hud_y = 30
            cv2.putText(output, f"FPS: {fps_avg:.1f} | Inference: {total_time:.0f}ms | Persons: {person_count}",
                       (10, hud_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

            hud_y += 30
            if too_close_count > 0:
                cv2.putText(output, f"TOO CLOSE: {too_close_count}/{person_count}",
                           (10, hud_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
            elif border_count > 0:
                cv2.putText(output, f"Very Close: {border_count}/{person_count}",
                           (10, hud_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 255), 2, cv2.LINE_AA)
            elif sitting_count > 0:
                cv2.putText(output, f"Standing: {standing_count} | Sitting: {sitting_count}",
                           (10, hud_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2, cv2.LINE_AA)

            # Write video frame
            if writer is not None:
                writer.write(output)

            # Show window
            if not args.no_show:
                cv2.imshow("YOLOv8-Pose Distance Estimation", output)
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q') or key == 27:
                    print("\nExiting...")
                    break
                elif key == ord('s'):
                    filepath = save_screenshot(output, args.screenshot_dir)
                    print(f"Screenshot saved: {filepath}")
                elif key == ord('c') and args.calibrate and person_count > 0:
                    # Calibrate using first non-occluded person
                    for person in person_detections:
                        if not person['occlusion']['is_occluded']:
                            calibrated_fl = depth_estimator.calibrate_focal_length(
                                person['bbox'], args.calibrate_distance
                            )
                            if calibrated_fl is not None:
                                depth_estimator.focal_length = calibrated_fl
                                print(f"\nFocal length calibrated: {depth_estimator.focal_length:.1f}px")
                                break
                elif key == ord('k'):
                    draw_keypoints = not draw_keypoints
                    print(f"Keypoints: {'ON' if draw_keypoints else 'OFF'}")

    finally:
        cap.release()
        if writer is not None:
            writer.release()
            print(f"Video saved to: {args.save_video}")
        if not args.no_show:
            cv2.destroyAllWindows()

        # Print statistics
        print(f"\n{'='*70}")
        print(f"Session Statistics:")
        print(f"  Total frames: {frame_count}")
        print(f"  Average FPS: {fps_avg:.2f}")
        print(f"  Average inference time: {avg_pose_ms:.2f}ms")
        print(f"  Final focal length: {depth_estimator.focal_length:.1f}px")
        print(f"{'='*70}\n")


def main():
    args = parse_args()

    print("Initializing YOLOv8-Pose with Depth Estimation...")

    # Initialize depth estimator
    depth_estimator = YOLOv8PoseDepth(
        pose_model_path=args.pose_model,
        person_height_cm=args.person_height,
        focal_length=args.focal_length,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        input_size=args.pose_size,
        min_keypoints=args.min_keypoints,
        border_threshold=args.border_threshold,
        border_distance_factor=args.border_distance_factor,
        kpt_threshold=args.kpt_threshold
    )

    print("\nModel loaded successfully!")

    # Run appropriate mode
    if args.image:
        run_image(args, depth_estimator)
    elif args.webcam:
        cap = cv2.VideoCapture(args.cam_id)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open webcam {args.cam_id}")
        run_video_stream(args, cap, f"Webcam {args.cam_id}", depth_estimator)
    elif args.video:
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {args.video}")
        run_video_stream(args, cap, args.video, depth_estimator)
    else:
        print("Error: Specify --image, --webcam, or --video")
        print("\nExamples:")
        print("  Image:   python run_yolov8poseDepth.py --image test.jpg")
        print("  Webcam:  python run_yolov8poseDepth.py --webcam")
        print("  Video:   python run_yolov8poseDepth.py --video input.mp4 --save-video output.mp4")


if __name__ == "__main__":
    main()
