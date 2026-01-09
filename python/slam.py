import cv2
import numpy as np
import argparse


class CameraMotionDetector:
    def __init__(self, threshold=1.0, flow_method='farneback'):
        """
        Initialize camera motion detector.

        Args:
            threshold: Motion threshold - higher values = less sensitive
            flow_method: 'farneback' or 'lucas-kanade'
        """
        self.threshold = threshold
        self.flow_method = flow_method
        self.prev_gray = None

        # Continuous motion detection
        self.motion_frame_count = 0
        self.required_motion_frames = 3  # Require 3 consecutive frames of motion

        # For Lucas-Kanade method
        self.lk_params = dict(winSize=(15, 15),
                             maxLevel=2,
                             criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        # Feature detection parameters
        self.feature_params = dict(maxCorners=100,
                                  qualityLevel=0.3,
                                  minDistance=7,
                                  blockSize=7)

        self.prev_points = None

    def calculate_optical_flow_farneback(self, gray1, gray2):
        """Calculate dense optical flow using Farneback method."""
        flow = cv2.calcOpticalFlowFarneback(
            gray1, gray2, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )

        # Calculate magnitude and angle
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        # Calculate average motion
        avg_motion = np.mean(mag)
        max_motion = np.max(mag)

        return avg_motion, max_motion, flow

    def calculate_optical_flow_lucas_kanade(self, gray1, gray2):
        """Calculate sparse optical flow using Lucas-Kanade method."""
        if self.prev_points is None:
            self.prev_points = cv2.goodFeaturesToTrack(gray1, mask=None, **self.feature_params)

        if self.prev_points is None or len(self.prev_points) == 0:
            return 0.0, 0.0, None

        # Calculate optical flow
        next_points, status, error = cv2.calcOpticalFlowPyrLK(
            gray1, gray2, self.prev_points, None, **self.lk_params
        )

        if next_points is None:
            return 0.0, 0.0, None

        # Select good points
        good_new = next_points[status == 1]
        good_old = self.prev_points[status == 1]

        if len(good_new) == 0:
            self.prev_points = cv2.goodFeaturesToTrack(gray2, mask=None, **self.feature_params)
            return 0.0, 0.0, None

        # Calculate motion vectors
        motion_vectors = good_new - good_old
        distances = np.sqrt(np.sum(motion_vectors**2, axis=1))

        avg_motion = np.mean(distances)
        max_motion = np.max(distances)

        # Update points
        self.prev_points = cv2.goodFeaturesToTrack(gray2, mask=None, **self.feature_params)

        return avg_motion, max_motion, (good_old, good_new, status)

    def detect_motion(self, frame):
        """
        Detect if camera is moving based on optical flow.

        Args:
            frame: Current video frame

        Returns:
            is_moving: Boolean indicating if camera is moving
            avg_motion: Average motion magnitude
            max_motion: Maximum motion magnitude
            visualization: Frame with motion visualization
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.prev_gray is None:
            self.prev_gray = gray
            return False, 0.0, 0.0, frame

        # Calculate optical flow
        if self.flow_method == 'farneback':
            avg_motion, max_motion, flow = self.calculate_optical_flow_farneback(self.prev_gray, gray)
            vis_frame = self.visualize_flow_farneback(frame, flow)
        else:
            avg_motion, max_motion, tracking_data = self.calculate_optical_flow_lucas_kanade(self.prev_gray, gray)
            vis_frame = self.visualize_flow_lucas_kanade(frame, tracking_data)

        # Check if motion detected in this frame
        motion_detected = avg_motion > self.threshold

        # Update continuous motion counter
        if motion_detected:
            self.motion_frame_count += 1
        else:
            self.motion_frame_count = 0

        # Camera is considered moving only after continuous motion for required frames
        is_moving = self.motion_frame_count >= self.required_motion_frames

        self.prev_gray = gray

        return is_moving, avg_motion, max_motion, vis_frame

    def visualize_flow_farneback(self, frame, flow):
        """Visualize dense optical flow."""
        if flow is None:
            return frame

        vis = frame.copy()
        h, w = flow.shape[:2]
        step = 16

        # Draw flow vectors
        for y in range(0, h, step):
            for x in range(0, w, step):
                fx, fy = flow[y, x]
                if abs(fx) > 0.5 or abs(fy) > 0.5:
                    cv2.arrowedLine(vis, (x, y),
                                   (int(x + fx), int(y + fy)),
                                   (0, 255, 0), 1, tipLength=0.3)

        return vis

    def visualize_flow_lucas_kanade(self, frame, tracking_data):
        """Visualize sparse optical flow."""
        if tracking_data is None:
            return frame

        vis = frame.copy()
        good_old, good_new, status = tracking_data

        # Draw tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            vis = cv2.line(vis, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
            vis = cv2.circle(vis, (int(a), int(b)), 5, (0, 0, 255), -1)

        return vis

    def reset(self):
        """Reset the detector state."""
        self.prev_gray = None
        self.prev_points = None
        self.motion_frame_count = 0


def main():
    parser = argparse.ArgumentParser(description='Detect camera motion using optical flow')
    parser.add_argument('--source', type=str, default='0',
                       help='Video source: camera index (0, 1, ...) or video file path')
    parser.add_argument('--threshold', type=float, default=2.0,
                       help='Motion threshold (higher = less sensitive)')
    parser.add_argument('--method', type=str, default='farneback',
                       choices=['farneback', 'lucas-kanade'],
                       help='Optical flow method')
    parser.add_argument('--output', type=str, default=None,
                       help='Output video file path (optional)')

    args = parser.parse_args()

    # Open video source
    if args.source.isdigit():
        cap = cv2.VideoCapture(int(args.source))
    else:
        cap = cv2.VideoCapture(args.source)

    if not cap.isOpened():
        print(f"Error: Could not open video source {args.source}")
        return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30

    # Initialize motion detector
    detector = CameraMotionDetector(threshold=args.threshold, flow_method=args.method)

    # Setup video writer if output path is provided
    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    print(f"Starting camera motion detection...")
    print(f"Method: {args.method}, Threshold: {args.threshold}")
    print("Press 'q' to quit, 'r' to reset detector")

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Detect motion
        is_moving, avg_motion, max_motion, vis_frame = detector.detect_motion(frame)

        # Display status
        status_text = "MOVING" if is_moving else "STATIC"
        color = (0, 0, 255) if is_moving else (0, 255, 0)

        cv2.putText(vis_frame, f"Status: {status_text}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(vis_frame, f"Avg Motion: {avg_motion:.2f}", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(vis_frame, f"Max Motion: {max_motion:.2f}", (10, 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(vis_frame, f"Frame: {frame_count}", (10, 130),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Show frame
        cv2.imshow('Camera Motion Detection', vis_frame)

        # Write to output video
        if writer:
            writer.write(vis_frame)

        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            detector.reset()
            print("Detector reset")

    # Cleanup
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    print(f"\nProcessed {frame_count} frames")
    if args.output:
        print(f"Output saved to: {args.output}")


if __name__ == '__main__':
    main()
