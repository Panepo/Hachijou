# run_slam.py
import argparse
import time
import cv2
import os
from datetime import datetime
from slam import CameraMotionDetector


def parse_args():
    parser = argparse.ArgumentParser(description="Run Camera Motion Detection (SLAM) on image sequence, webcam or video.")

    # Detection settings
    parser.add_argument("--threshold", type=float, default=0.5,
                       help="Motion threshold (higher = less sensitive, default: 0.5).")
    parser.add_argument("--method", type=str, default="lucas-kanade",
                       choices=["farneback", "lucas-kanade"],
                       help="Optical flow method (default: lucas-kanade).")

    # Input settings
    parser.add_argument("--video", type=str, help="Path to input video file.")
    parser.add_argument("--webcam", action="store_true", help="Use webcam as input.")
    parser.add_argument("--cam-id", type=int, default=0, help="Webcam device ID (default: 0).")

    # Output settings
    parser.add_argument("--save-video", type=str, help="Path to save output video.")
    parser.add_argument("--screenshot-dir", type=str, default="./screenshots",
                       help="Directory to save screenshots (default: ./screenshots).")
    parser.add_argument("--no-show", action="store_true", help="Don't display video window.")
    parser.add_argument("--flip", action="store_true", help="Flip webcam horizontally.")

    return parser.parse_args()


def run_video(args):
    """Run camera motion detection on video or webcam"""
    # Open video source
    if args.webcam:
        cap = cv2.VideoCapture(args.cam_id)
        print(f"Opening webcam (device {args.cam_id})...")
    elif args.video:
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

    # Initialize motion detector
    detector = CameraMotionDetector(threshold=args.threshold, flow_method=args.method)
    # Setup video writer if save path is provided
    writer = None
    if args.save_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(args.save_video, fourcc, fps, (width, height))
        print(f"Saving output to: {args.save_video}")

    # Create screenshot directory if needed
    if not os.path.exists(args.screenshot_dir):
        os.makedirs(args.screenshot_dir)

    print("\nControls:")
    print("  'q' - Quit")
    print("  'r' - Reset detector")
    print("  's' - Save screenshot")
    print("  SPACE - Pause/Resume")
    print()

    frame_count = 0
    paused = False
    start_time = time.time()

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("\nEnd of video stream")
                break

            frame_count += 1

            # Flip frame if requested
            if args.flip:
                frame = cv2.flip(frame, 1)

            # Detect motion
            is_moving, avg_motion, max_motion, vis_frame = detector.detect_motion(frame)

            # Display status
            status_text = "MOVING" if is_moving else "STATIC"
            color = (0, 0, 255) if is_moving else (0, 255, 0)

            cv2.putText(vis_frame, f"Status: {status_text}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.putText(vis_frame, f"Method: {args.method}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(vis_frame, f"Avg Motion: {avg_motion:.2f}", (10, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(vis_frame, f"Max Motion: {max_motion:.2f}", (10, 130),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            #cv2.putText(vis_frame, f"Frame: {frame_count}", (10, 160),
            #           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Calculate and display FPS
            elapsed = time.time() - start_time
            if elapsed > 0:
                current_fps = frame_count / elapsed
                cv2.putText(vis_frame, f"FPS: {current_fps:.1f}", (10, 190),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Show frame
            if not args.no_show:
                cv2.imshow('Camera Motion Detection (SLAM)', vis_frame)
        else:
            # Paused - just show the last frame
            if not args.no_show:
                cv2.imshow('Camera Motion Detection (SLAM)', vis_frame)

        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\nQuitting...")
            break
        elif key == ord('r'):
            detector.reset()
            print("Detector reset")
        elif key == ord('s'):
            # Save screenshot
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            screenshot_path = os.path.join(args.screenshot_dir, f"slam_{timestamp}.jpg")
            cv2.imwrite(screenshot_path, vis_frame)
            print(f"Screenshot saved: {screenshot_path}")
        elif key == ord(' '):
            # Pause/Resume
            paused = not paused
            status = "PAUSED" if paused else "RESUMED"
            print(f"{status}")

    # Cleanup
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    # Print statistics
    elapsed = time.time() - start_time
    avg_fps = frame_count / elapsed if elapsed > 0 else 0
    print(f"\nProcessing complete:")
    print(f"  Total frames: {frame_count}")
    print(f"  Time elapsed: {elapsed:.2f} seconds")
    print(f"  Average FPS: {avg_fps:.1f}")
    if args.save_video:
        print(f"  Output saved to: {args.save_video}")


def main():
    args = parse_args()

    # Run video/webcam mode
    if args.webcam or args.video:
        run_video(args)
    else:
        print("Error: Please specify --webcam or --video <file>")
        print("Example usage:")
        print("  python run_slam.py --webcam")
        print("  python run_slam.py --video ./video.mp4")
        print("  python run_slam.py --webcam --save-video ./output/slam_output.mp4")


if __name__ == "__main__":
    main()
