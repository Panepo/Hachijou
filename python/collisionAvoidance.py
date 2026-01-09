import cv2
import numpy as np
from slam import CameraMotionDetector
from depthv2 import DepthAnythingV2


class CollisionAvoidance:
    """
    Surround monitoring system that detects objects too close to the camera.

    Uses SLAM for motion detection and Depth Anything V2 for depth estimation.
    Highlights areas where depth is less than the specified threshold.
    """

    def __init__(self,
                 depth_model_path,
                 collision_threshold=3.0,
                 motion_threshold=1.0,
                 flow_method='lucas-kanade',
                 input_size=(518, 518)):
        """
        Initialize collision avoidance system.

        Args:
            depth_model_path (str): Path to depth estimation ONNX model
            collision_threshold (float): Depth threshold for collision warning (default: 3.0)
            motion_threshold (float): Motion detection threshold (default: 1.0)
            flow_method (str): Optical flow method - 'farneback' or 'lucas-kanade'
            input_size (tuple): Depth model input size
        """
        # Initialize motion detector (SLAM)
        self.motion_detector = CameraMotionDetector(
            threshold=motion_threshold,
            flow_method=flow_method
        )

        # Initialize depth estimator
        self.depth_estimator = DepthAnythingV2(
            model_path=depth_model_path,
            input_size=input_size
        )

        self.collision_threshold = collision_threshold
        self.is_camera_moving = False
        self.current_depth_map = None

        # Statistics
        self.frames_processed = 0
        self.collision_warnings = 0

    def detect_motion(self, frame):
        """
        Detect if camera is moving.

        Args:
            frame: Input video frame

        Returns:
            tuple: (is_moving, avg_motion, max_motion, visualization)
        """
        is_moving, avg_motion, max_motion, vis_frame = self.motion_detector.detect_motion(frame)
        self.is_camera_moving = is_moving

        return is_moving, avg_motion, max_motion, vis_frame

    def estimate_depth(self, frame):
        """
        Estimate depth map for the current frame.

        Args:
            frame: Input video frame

        Returns:
            tuple: (depth_map, colored_depth_visualization)
        """
        depth_map, colored_depth = self.depth_estimator.predict_and_visualize(frame)
        self.current_depth_map = depth_map

        return depth_map, colored_depth

    def detect_collision_risk(self, depth_map):
        """
        Detect areas where objects are too close (depth > threshold).

        Args:
            depth_map: Depth map from depth estimation

        Returns:
            tuple: (collision_mask, has_collision, collision_percentage)
        """
        # Create binary mask for collision areas where depth > threshold
        # Higher depth values indicate closer objects in this depth model
        collision_mask = depth_map > self.collision_threshold

        # Calculate collision statistics
        total_pixels = collision_mask.size
        collision_pixels = np.sum(collision_mask)
        collision_percentage = (collision_pixels / total_pixels) * 100 if total_pixels > 0 else 0
        has_collision = collision_percentage > 1.0  # More than 1% of area

        return collision_mask, has_collision, collision_percentage

    def highlight_collision_areas(self, frame, collision_mask):
        """
        Highlight collision risk areas on the frame.

        Args:
            frame: Original video frame
            collision_mask: Binary mask of collision areas

        Returns:
            Frame with collision areas highlighted
        """
        # Create overlay
        overlay = frame.copy()

        # Resize masks to match frame size if needed
        if collision_mask.shape[:2] != frame.shape[:2]:
            collision_mask = cv2.resize(
                collision_mask.astype(np.uint8),
                (frame.shape[1], frame.shape[0]),
                interpolation=cv2.INTER_NEAREST
            ).astype(bool)

        # Highlight collision areas in red
        overlay[collision_mask] = [0, 0, 255]  # BGR: Red

        # Blend with original frame
        highlighted = cv2.addWeighted(frame, 0.6, overlay, 0.4, 0)

        # Draw contours around collision areas
        contours, _ = cv2.findContours(
            collision_mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        cv2.drawContours(highlighted, contours, -1, (0, 0, 255), 2)

        return highlighted

    def create_visualization(self, frame, depth_colored, collision_mask,
                           is_moving, avg_motion, has_collision, collision_percentage):
        """
        Create comprehensive visualization with all information.

        Args:
            frame: Original frame
            depth_colored: Colored depth visualization (not used)
            collision_mask: Collision risk mask
            is_moving: Whether camera is moving
            avg_motion: Average motion value
            has_collision: Whether collision risk detected
            collision_percentage: Percentage of frame with collision risk

        Returns:
            Visualization frame with collision highlights
        """
        # Highlight collision areas on original frame
        visualization = self.highlight_collision_areas(frame, collision_mask)

        # Add status information
        status_y = 30
        line_height = 35

        # Camera motion status
        motion_status = "MOVING" if is_moving else "WAITING (Static)"
        motion_color = (0, 255, 0) if is_moving else (0, 165, 255)  # Orange when static
        cv2.putText(visualization, f"Camera: {motion_status}", (10, status_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, motion_color, 2, cv2.LINE_AA)

        # Motion value
        cv2.putText(visualization, f"Motion: {avg_motion:.2f}", (10, status_y + line_height),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        # Collision warning
        if is_moving:
            warning_text = "COLLISION RISK!" if has_collision else "Safe"
            warning_color = (0, 0, 255) if has_collision else (0, 255, 0)
            cv2.putText(visualization, f"Status: {warning_text}", (10, status_y + line_height * 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, warning_color, 2, cv2.LINE_AA)

            # Collision percentage
            cv2.putText(visualization, f"Risk Area: {collision_percentage:.1f}%",
                       (10, status_y + line_height * 3),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        # Threshold info
        cv2.putText(visualization, f"Threshold: {self.collision_threshold}",
                   (10, status_y + line_height * 4),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

        return visualization

    def process_frame(self, frame):
        """
        Process a single frame through the complete pipeline.

        Args:
            frame: Input video frame

        Returns:
            tuple: (visualization, is_moving, has_collision, collision_percentage)
        """
        self.frames_processed += 1

        # Step 1: Detect camera motion
        is_moving, avg_motion, max_motion, motion_vis = self.detect_motion(frame)

        # Initialize default values
        has_collision = False
        collision_percentage = 0.0

        # Step 2: If camera is moving, estimate depth
        if is_moving:
            # Estimate depth
            depth_map, depth_colored = self.estimate_depth(frame)

            # Step 3: Detect collision risk
            collision_mask, has_collision, collision_percentage = self.detect_collision_risk(depth_map)

            # Track collision warnings
            if has_collision:
                self.collision_warnings += 1

            # Create visualization
            visualization = self.create_visualization(
                frame, depth_colored, collision_mask,
                is_moving, avg_motion, has_collision, collision_percentage
            )
        else:
            # Camera is static - just show motion detection on original frame
            visualization = frame.copy()

            # Add waiting message
            cv2.putText(visualization, "Waiting for camera movement...",
                       (visualization.shape[1]//2 - 250, visualization.shape[0]//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2, cv2.LINE_AA)

            # Add status
            cv2.putText(visualization, "Camera: WAITING (Static)", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2, cv2.LINE_AA)
            cv2.putText(visualization, f"Motion: {avg_motion:.2f}", (10, 65),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        return visualization, is_moving, has_collision, collision_percentage

    def reset(self):
        """Reset the collision avoidance system."""
        self.motion_detector.reset()
        self.current_depth_map = None
        self.is_camera_moving = False

    def get_statistics(self):
        """
        Get processing statistics.

        Returns:
            dict: Statistics dictionary
        """
        return {
            'frames_processed': self.frames_processed,
            'collision_warnings': self.collision_warnings,
            'warning_rate': (self.collision_warnings / max(1, self.frames_processed)) * 100
        }
