# run_collisionAvoidance.py
import argparse
import time
import cv2
import os
from datetime import datetime
from collisionAvoidance import CollisionAvoidance


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Collision Avoidance System - Detect objects too close to camera using SLAM and depth estimation."
    )

    # Model settings
    parser.add_argument("--model", type=str,
                       default="./models/depth_anything_v2_vits.onnx",
                       help="Path to depth estimation ONNX model (default: ./models/depth_anything_v2_vits.onnx)")
    parser.add_argument("--input-size", type=int, nargs=2, default=[518, 518],
                       help="Model input size (height width, default: 518 518)")

    # Detection settings
    parser.add_argument("--collision-threshold", type=float, default=3.0,
                       help="Depth threshold for collision warning (default: 3.0)")
    parser.add_argument("--motion-threshold", type=float, default=0.5,
                       help="Motion detection threshold (default: 0.5)")
    parser.add_argument("--flow-method", type=str, default="lucas-kanade",
                       choices=["farneback", "lucas-kanade"],
                       help="Optical flow method for motion detection (default: lucas-kanade)")

    # Input source
    parser.add_argument("--video", type=str, help="Path to input video file")
    parser.add_argument("--webcam", action="store_true", help="Use webcam as input")
    parser.add_argument("--cam-id", type=int, default=0, help="Webcam device ID (default: 0)")

    # Output settings
    parser.add_argument("--save-video", type=str, help="Path to save output video")
    parser.add_argument("--screenshot-dir", type=str, default="./screenshots",
                       help="Directory to save screenshots (default: ./screenshots)")
    parser.add_argument("--no-show", action="store_true", help="Don't display video window")
    parser.add_argument("--flip", action="store_true", help="Flip webcam horizontally")

    # Performance
    parser.add_argument("--max-fps", type=int, default=30,
                       help="Maximum FPS for processing (default: 30)")

    return parser.parse_args()


def save_screenshot(image, screenshot_dir, prefix="collision_avoidance"):
    """
    Save a screenshot with timestamp.

    Args:
        image: Image to save
        screenshot_dir: Directory to save screenshots
        prefix: Filename prefix

    Returns:
        str: Path to saved screenshot
    """
    os.makedirs(screenshot_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{timestamp}.jpg"
    filepath = os.path.join(screenshot_dir, filename)

    cv2.imwrite(filepath, image)

    return filepath


def main():
    """Main function to run collision avoidance system."""
    args = parse_args()

    # Validate model file
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        print("Please specify the correct model path using --model argument")
        return

    # Open video source
    if args.webcam:
        cap = cv2.VideoCapture(args.cam_id)
        print(f"Opening webcam (device {args.cam_id})...")
    elif args.video:
        if not os.path.exists(args.video):
            print(f"Error: Video file not found: {args.video}")
            return
        cap = cv2.VideoCapture(args.video)
        print(f"Opening video file: {args.video}")
    else:
        print("Error: Please specify --webcam or --video")
        return

    if not cap.isOpened():
        print("Error: Could not open video source")
        return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30

    print(f"Video resolution: {width}x{height} @ {fps} FPS")

    # Initialize collision avoidance system
    print("\nInitializing Collision Avoidance System...")
    print(f"Model: {args.model}")
    print(f"Collision Threshold: {args.collision_threshold}")
    print(f"Motion Threshold: {args.motion_threshold}")
    print(f"Flow Method: {args.flow_method}")

    collision_system = CollisionAvoidance(
        depth_model_path=args.model,
        collision_threshold=args.collision_threshold,
        motion_threshold=args.motion_threshold,
        flow_method=args.flow_method,
        input_size=tuple(args.input_size)
    )

    # Setup video writer if save path is provided
    writer = None
    if args.save_video:
        # Output will be single frame with highlights
        output_width = width
        output_height = height
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(args.save_video, fourcc, fps, (output_width, output_height))
        print(f"Saving output to: {args.save_video}")

    # Create screenshot directory
    os.makedirs(args.screenshot_dir, exist_ok=True)

    # Display controls
    print("\n" + "="*60)
    print("COLLISION AVOIDANCE SYSTEM - CONTROLS")
    print("="*60)
    print("  'q' or ESC  - Quit")
    print("  'r'         - Reset detector")
    print("  's'         - Save screenshot")
    print("  SPACE       - Pause/Resume")
    print("  '+'/'-'     - Adjust collision threshold")
    print("="*60)
    print()

    # Processing variables
    frame_count = 0
    paused = False
    start_time = time.time()
    processing_times = []

    # FPS limiting
    frame_delay = 1.0 / args.max_fps if args.max_fps > 0 else 0
    last_frame_time = time.time()

    try:
        while True:
            if not paused:
                # FPS limiting
                current_time = time.time()
                elapsed = current_time - last_frame_time
                if elapsed < frame_delay:
                    time.sleep(frame_delay - elapsed)
                last_frame_time = time.time()

                # Read frame
                ret, frame = cap.read()
                if not ret:
                    print("\nEnd of video stream")
                    break

                frame_count += 1

                # Flip frame if requested
                if args.flip:
                    frame = cv2.flip(frame, 1)

                # Process frame
                frame_start_time = time.time()

                visualization, is_moving, has_collision, collision_percentage = \
                    collision_system.process_frame(frame)

                frame_process_time = time.time() - frame_start_time
                processing_times.append(frame_process_time)

                # Calculate FPS
                if len(processing_times) > 30:
                    processing_times.pop(0)
                avg_process_time = sum(processing_times) / len(processing_times)
                current_fps = 1.0 / avg_process_time if avg_process_time > 0 else 0

                # Add FPS counter to visualization
                cv2.putText(visualization, f"FPS: {current_fps:.1f}",
                           (visualization.shape[1] - 150, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(visualization, f"Frame: {frame_count}",
                           (visualization.shape[1] - 150, 65),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

                # Display warning if collision detected
                if is_moving and has_collision:
                    # Add blinking warning indicator
                    if frame_count % 10 < 5:  # Blink every 10 frames
                        warning_text = "!!! WARNING: OBJECT TOO CLOSE !!!"
                        text_size = cv2.getTextSize(warning_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)[0]
                        text_x = (visualization.shape[1] - text_size[0]) // 2
                        text_y = visualization.shape[0] - 30

                        # Draw background
                        cv2.rectangle(visualization,
                                    (text_x - 10, text_y - text_size[1] - 10),
                                    (text_x + text_size[0] + 10, text_y + 10),
                                    (0, 0, 255), -1)
                        # Draw text
                        cv2.putText(visualization, warning_text, (text_x, text_y),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3, cv2.LINE_AA)

                # Show visualization
                if not args.no_show:
                    cv2.imshow('Collision Avoidance System', visualization)

                # Write to output video
                if writer:
                    writer.write(visualization)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q') or key == 27:  # 'q' or ESC
                print("\nQuitting...")
                break
            elif key == ord('r'):
                collision_system.reset()
                print("System reset")
            elif key == ord('s'):
                if not paused:
                    filepath = save_screenshot(visualization, args.screenshot_dir)
                    print(f"Screenshot saved: {filepath}")
            elif key == ord(' '):  # SPACE
                paused = not paused
                print("Paused" if paused else "Resumed")
            elif key == ord('+') or key == ord('='):
                collision_system.collision_threshold += 0.5
                print(f"Collision threshold increased to: {collision_system.collision_threshold}")
            elif key == ord('-') or key == ord('_'):
                collision_system.collision_threshold = max(0.5, collision_system.collision_threshold - 0.5)
                print(f"Collision threshold decreased to: {collision_system.collision_threshold}")

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")

    finally:
        # Cleanup
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()

        # Print statistics
        elapsed_time = time.time() - start_time
        stats = collision_system.get_statistics()

        print("\n" + "="*60)
        print("PROCESSING STATISTICS")
        print("="*60)
        print(f"Total frames processed: {frame_count}")
        print(f"Frames analyzed: {stats['frames_processed']}")
        print(f"Collision warnings: {stats['collision_warnings']}")
        print(f"Warning rate: {stats['warning_rate']:.2f}%")
        print(f"Processing time: {elapsed_time:.2f} seconds")
        print(f"Average FPS: {frame_count / elapsed_time:.2f}")
        print("="*60)

        if args.save_video:
            print(f"\nOutput video saved to: {args.save_video}")


if __name__ == "__main__":
    main()
