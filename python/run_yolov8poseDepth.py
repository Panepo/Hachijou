# run_yolov8poseDepth.py
import argparse
import time
import math
import cv2
import numpy as np
import os
from datetime import datetime
from yolov8pose import YOLOv8PoseONNX


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run YOLOv8-Pose for person distance estimation with occlusion detection based on keypoints."
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
                       help="Pixel threshold for detecting if bbox is on border (default: 50px).")
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


def estimate_focal_length(image_width, image_height, fov_horizontal=60.0):
    """
    Estimate focal length based on typical webcam field of view

    Args:
        image_width: Image width in pixels
        image_height: Image height in pixels
        fov_horizontal: Horizontal field of view in degrees (default: 60)

    Returns:
        float: Estimated focal length in pixels
    """
    focal_length_x = (image_width / 2) / math.tan(math.radians(fov_horizontal / 2))
    aspect_ratio = image_width / image_height
    fov_vertical = 2 * math.atan(math.tan(math.radians(fov_horizontal / 2)) / aspect_ratio)
    focal_length_y = (image_height / 2) / math.tan(fov_vertical / 2)
    focal_length = (focal_length_x + focal_length_y) / 2
    return focal_length


def save_screenshot(image, screenshot_dir="./screenshots"):
    """Save a screenshot with timestamp"""
    os.makedirs(screenshot_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"pose_depth_screenshot_{timestamp}.jpg"
    filepath = os.path.join(screenshot_dir, filename)
    cv2.imwrite(filepath, image)
    return filepath


def get_keypoint_span_height(keypoints, kpt_threshold=0.5):
    """
    Calculate the vertical span of visible keypoints
    This represents the visible height of the person in pixels

    Args:
        keypoints: List of keypoints [[x, y, conf], ...]
        kpt_threshold: Confidence threshold

    Returns:
        float: Vertical span in pixels (max_y - min_y)
    """
    visible_y = [kpt[1] for kpt in keypoints if kpt[2] > kpt_threshold]

    if len(visible_y) < 2:
        return 0.0

    return max(visible_y) - min(visible_y)


def is_bbox_on_border(bbox, frame_shape, border_threshold=50):
    """
    Detect if bounding box touches or is near the frame border
    This indicates the person is very close to the camera

    Args:
        bbox: [x1, y1, x2, y2]
        frame_shape: (height, width) of the frame
        border_threshold: Pixel distance from edge to consider "on border"

    Returns:
        dict: {
            'on_border': bool,
            'edges': list of strings indicating which edges ('top', 'bottom', 'left', 'right')
        }
    """
    x1, y1, x2, y2 = bbox
    frame_height, frame_width = frame_shape[:2]

    edges = []

    # Check each edge
    if x1 <= border_threshold:
        edges.append('left')
    if x2 >= frame_width - border_threshold:
        edges.append('right')
    if y1 <= border_threshold:
        edges.append('top')
    if y2 >= frame_height - border_threshold:
        edges.append('bottom')

    return {
        'on_border': len(edges) > 0,
        'edges': edges
    }


def detect_standing_posture(keypoints, kpt_threshold=0.5):
    """
    Detect if person is in standing posture based on body keypoint alignment

    COCO keypoint indices:
    0: nose, 1-2: eyes, 3-4: ears, 5-6: shoulders, 7-8: elbows, 9-10: wrists
    11-12: hips, 13-14: knees, 15-16: ankles

    Args:
        keypoints: List of 17 keypoints [[x, y, conf], ...]
        kpt_threshold: Keypoint confidence threshold

    Returns:
        dict: Standing posture analysis with keys:
            - is_standing: bool
            - confidence: float (0-1, how confident we are in the classification)
            - reason: str (explanation)
            - vertical_alignment: float (0-1, how vertical the body is)
    """
    # Get key body points
    left_shoulder = keypoints[5] if len(keypoints) > 5 else [0, 0, 0]
    right_shoulder = keypoints[6] if len(keypoints) > 6 else [0, 0, 0]
    left_hip = keypoints[11] if len(keypoints) > 11 else [0, 0, 0]
    right_hip = keypoints[12] if len(keypoints) > 12 else [0, 0, 0]
    left_knee = keypoints[13] if len(keypoints) > 13 else [0, 0, 0]
    right_knee = keypoints[14] if len(keypoints) > 14 else [0, 0, 0]
    left_ankle = keypoints[15] if len(keypoints) > 15 else [0, 0, 0]
    right_ankle = keypoints[16] if len(keypoints) > 16 else [0, 0, 0]

    # Check visibility of critical joints
    shoulder_visible = (left_shoulder[2] > kpt_threshold or right_shoulder[2] > kpt_threshold)
    hip_visible = (left_hip[2] > kpt_threshold or right_hip[2] > kpt_threshold)
    knee_visible = (left_knee[2] > kpt_threshold or right_knee[2] > kpt_threshold)
    ankle_visible = (left_ankle[2] > kpt_threshold or right_ankle[2] > kpt_threshold)

    # Can't determine if critical points are missing
    if not (shoulder_visible and hip_visible):
        return {
            'is_standing': False,
            'confidence': 0.0,
            'reason': 'insufficient_keypoints',
            'vertical_alignment': 0.0
        }

    # Calculate average positions for each body part
    shoulder_y = (left_shoulder[1] if left_shoulder[2] > kpt_threshold else 0) + \
                 (right_shoulder[1] if right_shoulder[2] > kpt_threshold else 0)
    shoulder_count = (1 if left_shoulder[2] > kpt_threshold else 0) + \
                     (1 if right_shoulder[2] > kpt_threshold else 0)
    shoulder_y /= max(shoulder_count, 1)

    hip_y = (left_hip[1] if left_hip[2] > kpt_threshold else 0) + \
            (right_hip[1] if right_hip[2] > kpt_threshold else 0)
    hip_count = (1 if left_hip[2] > kpt_threshold else 0) + \
                (1 if right_hip[2] > kpt_threshold else 0)
    hip_y /= max(hip_count, 1)

    knee_y = (left_knee[1] if left_knee[2] > kpt_threshold else 0) + \
             (right_knee[1] if right_knee[2] > kpt_threshold else 0)
    knee_count = (1 if left_knee[2] > kpt_threshold else 0) + \
                 (1 if right_knee[2] > kpt_threshold else 0)
    knee_y /= max(knee_count, 1) if knee_count > 0 else 1

    ankle_y = (left_ankle[1] if left_ankle[2] > kpt_threshold else 0) + \
              (right_ankle[1] if right_ankle[2] > kpt_threshold else 0)
    ankle_count = (1 if left_ankle[2] > kpt_threshold else 0) + \
                  (1 if right_ankle[2] > kpt_threshold else 0)
    ankle_y /= max(ankle_count, 1) if ankle_count > 0 else 1

    # Calculate torso height (shoulder to hip)
    torso_height = abs(hip_y - shoulder_y)

    # Calculate vertical alignment metrics
    vertical_alignment = 0.0

    if knee_visible and ankle_visible and torso_height > 0:
        # Full body visible - check vertical progression
        # In standing: shoulder -> hip -> knee -> ankle should progress downward
        hip_knee_height = abs(knee_y - hip_y)
        knee_ankle_height = abs(ankle_y - knee_y)

        # Check if body segments progress vertically (Y increases going down)
        shoulder_to_hip_ok = hip_y > shoulder_y  # hip below shoulder
        hip_to_knee_ok = knee_y > hip_y  # knee below hip
        knee_to_ankle_ok = ankle_y > knee_y  # ankle below knee

        # Standing person has roughly proportional body segments
        # Torso ~ 40%, Upper leg ~ 25%, Lower leg ~ 25% of body height
        total_height = ankle_y - shoulder_y

        if total_height > 0:
            torso_ratio = torso_height / total_height
            leg_ratio = (hip_knee_height + knee_ankle_height) / total_height

            # Standing: torso is roughly 35-50% of total height
            # Legs (hip to ankle) are roughly 50-65% of total height
            is_proportional = (0.30 < torso_ratio < 0.55) and (0.45 < leg_ratio < 0.70)

            # Calculate vertical alignment score
            alignment_checks = [shoulder_to_hip_ok, hip_to_knee_ok, knee_to_ankle_ok, is_proportional]
            vertical_alignment = sum(alignment_checks) / len(alignment_checks)

            # Strong standing indicators
            if vertical_alignment >= 0.80:
                return {
                    'is_standing': True,
                    'confidence': vertical_alignment,
                    'reason': 'vertical_full_body',
                    'vertical_alignment': vertical_alignment
                }
            else:
                return {
                    'is_standing': False,
                    'confidence': 1.0 - vertical_alignment,
                    'reason': 'sitting_or_lying',
                    'vertical_alignment': vertical_alignment
                }

    elif knee_visible and torso_height > 0:
        # Knees visible but not ankles
        hip_knee_height = abs(knee_y - hip_y)

        shoulder_to_hip_ok = hip_y > shoulder_y
        hip_to_knee_ok = knee_y > hip_y

        # Check proportions
        upper_body_height = knee_y - shoulder_y
        if upper_body_height > 0:
            torso_ratio = torso_height / upper_body_height

            # Standing: torso should be about 40-60% of visible height (shoulder to knee)
            is_proportional = 0.35 < torso_ratio < 0.65

            alignment_checks = [shoulder_to_hip_ok, hip_to_knee_ok, is_proportional]
            vertical_alignment = sum(alignment_checks) / len(alignment_checks)

            if vertical_alignment >= 0.66:
                return {
                    'is_standing': True,
                    'confidence': vertical_alignment * 0.8,  # Lower confidence without ankles
                    'reason': 'vertical_upper_body',
                    'vertical_alignment': vertical_alignment
                }

    # Default to not standing if we can't determine
    return {
        'is_standing': False,
        'confidence': 0.6,
        'reason': 'unclear_posture',
        'vertical_alignment': vertical_alignment
    }


def detect_occlusion(keypoints, kpt_threshold=0.5):
    """
    Detect if person is occluded based on missing keypoints

    Args:
        keypoints: List of 17 keypoints [[x, y, conf], ...]
        kpt_threshold: Keypoint confidence threshold

    Returns:
        dict: Occlusion analysis with keys:
            - is_occluded: bool
            - visible_keypoints: int
            - missing_keypoints: list of indices
            - occlusion_type: str ('none', 'upper_body', 'lower_body', 'partial')
            - visible_upper: int
            - visible_lower: int
    """
    # Define keypoint groups (COCO format, 0-indexed)
    upper_body = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # nose to wrists
    lower_body = [11, 12, 13, 14, 15, 16]  # hips to ankles

    visible_keypoints = []
    missing_keypoints = []
    visible_upper = 0
    visible_lower = 0

    for i, (x, y, conf) in enumerate(keypoints):
        if conf > kpt_threshold:
            visible_keypoints.append(i)
            if i in upper_body:
                visible_upper += 1
            if i in lower_body:
                visible_lower += 1
        else:
            missing_keypoints.append(i)

    # Determine occlusion type
    num_visible = len(visible_keypoints)
    is_occluded = False
    occlusion_type = 'none'

    # Generally too few keypoints
    if num_visible < 10:
        is_occluded = True
        occlusion_type = 'partial'

    # Missing lower body keypoints (e.g., sitting, behind desk)
    elif visible_lower < 4 and visible_upper >= 3:
        is_occluded = True
        occlusion_type = 'lower_body'

    # Missing upper body keypoints
    elif visible_upper < 5 and visible_lower >= 2:
        is_occluded = True
        occlusion_type = 'upper_body'

    return {
        'is_occluded': is_occluded,
        'visible_keypoints': num_visible,
        'missing_keypoints': missing_keypoints,
        'occlusion_type': occlusion_type,
        'visible_upper': visible_upper,
        'visible_lower': visible_lower
    }


def calculate_distance_from_bbox(bbox, person_height_cm, focal_length, occlusion_info=None, posture_info=None):
    """
    Calculate distance using bounding box height
    Distance = (Real Height × Focal Length) / Pixel Height

    Adjusts person height based on posture and occlusion:
    - If standing: use full height (person_height_cm)
    - If not standing (sitting/lying): use fixed reduced height
    - If lower body occluded: use half height (upper body only)
    - If upper body occluded: use half height (lower body only)

    Args:
        bbox: Bounding box [x1, y1, x2, y2]
        person_height_cm: Actual person height in cm (when standing)
        focal_length: Camera focal length in pixels
        occlusion_info: Occlusion detection dict (optional)
        posture_info: Posture detection dict (optional)

    Returns:
        float: Estimated distance in cm, or None if calculation fails
    """
    if focal_length is None or focal_length <= 0:
        return None

    # Get bbox height
    x1, y1, x2, y2 = bbox
    pixel_height = y2 - y1

    if pixel_height <= 0:
        return None

    # Adjust person height based on posture and occlusion
    effective_height = person_height_cm
    height_source = "full_standing"

    # First check posture (takes priority over occlusion heuristics)
    if posture_info is not None and posture_info['confidence'] > 0.5:
        if not posture_info['is_standing']:
            # Person is sitting or lying down - use fixed reduced height
            # Typical sitting height is about 55-65% of standing height
            effective_height = person_height_cm * 0.60
            height_source = "sitting_fixed"

    # If no clear posture detected, fall back to occlusion heuristics
    elif occlusion_info is not None:
        # Check if bounding box is near square (aspect ratio close to 1)
        bbox_width = x2 - x1
        bbox_height = y2 - y1
        aspect_ratio = bbox_width / bbox_height if bbox_height > 0 else 0
        is_near_square = 0.8 <= aspect_ratio <= 1.2

        if is_near_square and occlusion_info['occlusion_type'] == 'partial':
            # Near square bbox with partial occlusion suggests seated/crouched position
            effective_height = person_height_cm / 3
            height_source = "crouched"
        elif occlusion_info['occlusion_type'] == 'partial':
            # General partial occlusion, use three-quarters height
            effective_height = person_height_cm / 2.5
            height_source = "partial_occluded"
        elif occlusion_info['occlusion_type'] == 'lower_body':
            # Only upper body visible, use half height
            effective_height = person_height_cm / 2.0
            height_source = "upper_body_only"
        elif occlusion_info['occlusion_type'] == 'upper_body':
            # Only lower body visible, use half height
            effective_height = person_height_cm / 2.0
            height_source = "lower_body_only"

    distance = (effective_height * focal_length) / pixel_height
    return distance, effective_height, height_source


def calibrate_focal_length(bbox, person_height_cm, known_distance_cm):
    """
    Calibrate focal length based on known distance
    Focal Length = (Pixel Height × Known Distance) / Real Height

    Args:
        bbox: Bounding box [x1, y1, x2, y2]
        person_height_cm: Actual person height in cm
        known_distance_cm: Known distance to person in cm

    Returns:
        float: Calibrated focal length in pixels, or None if calculation fails
    """
    # Get bbox height
    x1, y1, x2, y2 = bbox
    pixel_height = y2 - y1

    if pixel_height <= 0:
        return None

    focal_length = (pixel_height * known_distance_cm) / person_height_cm
    return focal_length


def draw_person_with_info(image, detection, distance_cm, effective_height, height_source,
                          occlusion_info, posture_info, border_info,
                          pose_detector, draw_keypoints=True, draw_skeleton=True,
                          kpt_threshold=0.5):
    """
    Draw person with pose, distance, posture, and occlusion information

    Args:
        image: Image to draw on
        detection: Detection dict from YOLOv8-Pose
        distance_cm: Calculated distance in cm
        effective_height: Effective height used for calculation in cm
        height_source: Source of height estimation
        occlusion_info: Occlusion analysis dict
        posture_info: Posture detection dict
        border_info: Border detection dict
        pose_detector: YOLOv8PoseONNX instance
        draw_keypoints: Whether to draw keypoints
        draw_skeleton: Whether to draw skeleton
        kpt_threshold: Keypoint confidence threshold

    Returns:
        numpy.ndarray: Image with annotations
    """
    bbox = detection['bbox']
    confidence = detection['confidence']
    keypoints = detection['keypoints']

    x1, y1, x2, y2 = map(int, bbox)

    # Choose color based on proximity and occlusion status
    if border_info.get('too_close', False):
        box_color = (0, 0, 255)  # Red for extremely close (on border + few keypoints)
    elif border_info['on_border']:
        box_color = (0, 100, 255)  # Orange-red for very close (on border)
    elif occlusion_info['is_occluded']:
        box_color = (0, 165, 255)  # Orange for occluded
    elif posture_info and not posture_info['is_standing'] and posture_info['confidence'] > 0.5:
        box_color = (255, 165, 0)  # Blue for sitting/lying
    else:
        box_color = (0, 255, 0)  # Green for clear view

    # Draw bounding box
    cv2.rectangle(image, (x1, y1), (x2, y2), box_color, 2)

    # Draw skeleton
    if draw_skeleton:
        for sk in pose_detector.skeleton:
            kpt1_idx = sk[0] - 1
            kpt2_idx = sk[1] - 1

            if kpt1_idx >= len(keypoints) or kpt2_idx >= len(keypoints):
                continue

            kpt1 = keypoints[kpt1_idx]
            kpt2 = keypoints[kpt2_idx]

            if kpt1[2] > kpt_threshold and kpt2[2] > kpt_threshold:
                pt1 = (int(kpt1[0]), int(kpt1[1]))
                pt2 = (int(kpt2[0]), int(kpt2[1]))
                cv2.line(image, pt1, pt2, (255, 150, 0), 2)

    # Draw keypoints
    if draw_keypoints:
        for i, (kpt_x, kpt_y, kpt_conf) in enumerate(keypoints):
            if kpt_conf > kpt_threshold:
                pt = (int(kpt_x), int(kpt_y))
                cv2.circle(image, pt, 4, (0, 255, 0), -1)
                cv2.circle(image, pt, 5, (0, 0, 0), 1)
            elif kpt_conf > 0.1:  # Draw low-confidence keypoints differently
                pt = (int(kpt_x), int(kpt_y))
                cv2.circle(image, pt, 3, (100, 100, 100), -1)

    # Prepare labels
    labels = []
    labels.append(f"Person: {confidence:.2f}")
    labels.append(f"Keypoints: {occlusion_info['visible_keypoints']}/17")

    # Add posture information
    if posture_info and posture_info['confidence'] > 0.3:
        if posture_info['is_standing']:
            labels.append(f"Standing ({posture_info['confidence']:.0%})")
        else:
            posture_label = "Sitting" if posture_info['reason'] == 'sitting_or_lying' else "Not Standing"
            labels.append(f"{posture_label} ({posture_info['confidence']:.0%})")

    if distance_cm is not None:
        if distance_cm >= 100:
            dist_str = f"{distance_cm / 100:.2f}m"
        else:
            dist_str = f"{distance_cm:.0f}cm"

        # Show effective height if different from full height
        if height_source == "sitting_fixed":
            labels.append(f"Dist: {dist_str} (h={effective_height:.0f}cm)")
        else:
            labels.append(f"Distance: {dist_str}")
    else:
        labels.append("Distance: N/A")

    if border_info.get('too_close', False):
        labels.append("TOO CLOSE! (border+few kpts)")
    elif border_info['on_border']:
        edges_str = ','.join(border_info['edges'])
        labels.append(f"Very Close! ({edges_str})")
    elif occlusion_info['is_occluded']:
        if occlusion_info['occlusion_type'] in ['lower_body', 'upper_body']:
            labels.append(f"Half body ({occlusion_info['occlusion_type']})")
        else:
            labels.append(f"Occluded: {occlusion_info['occlusion_type']}")

    # Calculate label background size
    text_sizes = [cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1) for lbl in labels]
    max_width = max([size[0][0] for size in text_sizes])
    total_height = sum([size[0][1] + size[1] for size in text_sizes]) + 10

    # Draw label background
    cv2.rectangle(image, (x1, y1 - total_height - 5),
                 (x1 + max_width + 10, y1), box_color, -1)

    # Draw labels
    y_offset = y1 - 5
    for i in range(len(labels) - 1, -1, -1):
        cv2.putText(image, labels[i], (x1 + 5, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        y_offset -= text_sizes[i][0][1] + text_sizes[i][1] + 2

    return image


def process_frame(frame, pose_detector, person_height_cm,
                 focal_length, border_distance_factor,
                 border_threshold, min_keypoints, kpt_threshold,
                 draw_keypoints=True, draw_skeleton=True):
    """
    Process a single frame with pose detection

    Returns:
        tuple: (output_image, num_persons, pose_time, person_detections)
    """
    # Run pose detection
    t0 = time.time()
    detections = pose_detector.detect(frame)
    pose_time = (time.time() - t0) * 1000.0

    # Create output image
    output = frame.copy()
    person_detections = []

    # Process each detected person
    for detection in detections:
        bbox = detection['bbox']
        keypoints = detection['keypoints']

        # Detect if on border (very close to camera)
        border_info = is_bbox_on_border(bbox, frame.shape, border_threshold)

        # Detect occlusion
        occlusion_info = detect_occlusion(keypoints, kpt_threshold)

        # Detect standing posture
        posture_info = detect_standing_posture(keypoints, kpt_threshold)

        # Calculate distance with adjusted person height based on posture and occlusion
        result = calculate_distance_from_bbox(
            bbox, person_height_cm, focal_length, occlusion_info, posture_info
        )

        if result is not None:
            distance_cm, effective_height, height_source = result
        else:
            distance_cm, effective_height, height_source = None, person_height_cm, "unknown"

        # Adjust distance based on conditions
        adjusted_distance_cm = distance_cm
        is_too_close = False

        if distance_cm is not None:
            # Check if on border AND too few keypoints = extremely close
            if border_info['on_border'] and occlusion_info['visible_keypoints'] < min_keypoints:
                # Person is extremely close (cut off at edges with few visible keypoints)
                is_too_close = True
                adjusted_distance_cm = distance_cm * border_distance_factor

        # Update border info with too close flag
        border_info['too_close'] = is_too_close

        # Draw person with info
        output = draw_person_with_info(
            output, detection, adjusted_distance_cm, effective_height, height_source,
            occlusion_info, posture_info, border_info,
            pose_detector, draw_keypoints, draw_skeleton, kpt_threshold
        )

        # Store detection info
        person_detections.append({
            'bbox': bbox,
            'keypoints': keypoints,
            'distance_cm': distance_cm,
            'adjusted_distance_cm': adjusted_distance_cm,
            'effective_height': effective_height,
            'height_source': height_source,
            'confidence': detection['confidence'],
            'occlusion': occlusion_info,
            'posture': posture_info,
            'border': border_info,
            'bbox_height': bbox[3] - bbox[1]
        })

    return output, len(person_detections), pose_time, person_detections


def run_image(args, pose_detector, focal_length):
    """Process single image"""
    image = cv2.imread(args.image)
    if image is None:
        raise RuntimeError(f"Failed to load image: {args.image}")

    # Auto-estimate focal length if not provided
    if focal_length is None:
        h, w = image.shape[:2]
        focal_length = estimate_focal_length(w, h)
        print(f"Auto-estimated focal length: {focal_length:.1f}px (based on {w}x{h} and 60° FOV)")

    print(f"\nProcessing {args.image}...")

    # Process frame
    output, person_count, pose_time, person_detections = process_frame(
        image, pose_detector, args.person_height, focal_length,
        args.border_distance_factor,
        args.border_threshold, args.min_keypoints, args.kpt_threshold,
        not args.no_keypoints, not args.no_skeleton
    )

    # Print detection details
    print(f"\n{'='*70}")
    print(f"Detected {person_count} person(s)")
    print(f"Pose detection: {pose_time:.1f}ms")
    print(f"{'='*70}\n")

    for i, person in enumerate(person_detections, 1):
        print(f"Person {i}:")
        print(f"  Confidence: {person['confidence']:.3f}")
        print(f"  Visible keypoints: {person['occlusion']['visible_keypoints']}/17")

        # Display posture information
        if person['posture']['confidence'] > 0.3:
            posture_status = "Standing" if person['posture']['is_standing'] else "Sitting/Lying"
            print(f"  Posture: {posture_status} (confidence: {person['posture']['confidence']:.1%})")
            if person['posture']['vertical_alignment'] > 0:
                print(f"    - Vertical alignment: {person['posture']['vertical_alignment']:.1%}")

        if person['distance_cm'] is not None:
            dist_str = f"{person['distance_cm']/100:.2f}m" if person['distance_cm'] >= 100 else f"{person['distance_cm']:.0f}cm"

            # Show which height was used for calculation
            if person['height_source'] == 'sitting_fixed':
                print(f"  Calculated distance: {dist_str} (using sitting height: {person['effective_height']:.0f}cm)")
            elif person['height_source'] in ['lower_body_only', 'upper_body_only']:
                print(f"  Calculated distance: {dist_str} (using half-body height: {person['effective_height']:.0f}cm)")
            else:
                print(f"  Calculated distance: {dist_str} (using full height: {person['effective_height']:.0f}cm)")

            if person['border'].get('too_close', False):
                adj_dist_str = f"{person['adjusted_distance_cm']/100:.2f}m" if person['adjusted_distance_cm'] >= 100 else f"{person['adjusted_distance_cm']:.0f}cm"
                print(f"  Adjusted distance: {adj_dist_str} (TOO CLOSE - on border with {person['occlusion']['visible_keypoints']} keypoints)")
            elif person['border']['on_border']:
                adj_dist_str = f"{person['adjusted_distance_cm']/100:.2f}m" if person['adjusted_distance_cm'] >= 100 else f"{person['adjusted_distance_cm']:.0f}cm"
                edges_str = ', '.join(person['border']['edges'])
                print(f"  Adjusted distance: {adj_dist_str} (very close - on {edges_str} edge)")
            elif person['occlusion']['is_occluded']:
                adj_dist_str = f"{person['adjusted_distance_cm']/100:.2f}m" if person['adjusted_distance_cm'] >= 100 else f"{person['adjusted_distance_cm']:.0f}cm"
                print(f"  Adjusted distance: {adj_dist_str} (occluded: {person['occlusion']['occlusion_type']})")

        print(f"  Bbox height: {person['bbox_height']:.1f}px")
        print()

    # Save result
    import os
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


def run_video_stream(args, cap, source_name, pose_detector, focal_length):
    """Run on video stream (webcam or video file)"""
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_read = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # Auto-estimate focal length if not provided
    if focal_length is None:
        focal_length = estimate_focal_length(width, height)
        print(f"Auto-estimated focal length: {focal_length:.1f}px (based on {width}x{height} and 60° FOV)")
        if args.calibrate:
            print("Tip: Use calibration mode to improve accuracy")

    print(f"\n{'='*70}")
    print(f"Source: {source_name}")
    print(f"Resolution: {width}x{height} @ {fps_read:.1f} FPS")
    print(f"Person height assumption: {args.person_height}cm")
    print(f"Focal length: {focal_length:.1f}px")
    print(f"Occlusion detection: {args.min_keypoints}/17 keypoints minimum")
    print(f"  - Half-body occluded: uses 90cm height (auto-adjusted)")
    print(f"Border detection: {args.border_threshold}px threshold")
    print(f"Border distance factor: {args.border_distance_factor}x (very close)")
    if args.calibrate:
        print(f"Calibration: Press 'c' when at {args.calibrate_distance}cm")
    print(f"{'='*70}")
    print("\nControls:")
    print("  's' - Save screenshot")
    print("  'c' - Calibrate focal length (if --calibrate enabled)")
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

    current_focal_length = focal_length
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
            output, person_count, pose_time, person_detections = process_frame(
                frame, pose_detector, args.person_height,
                current_focal_length, args.border_distance_factor,
                args.border_threshold, args.min_keypoints,
                args.kpt_threshold, draw_keypoints, draw_skeleton
            )
            total_time = (time.time() - t0) * 1000.0

            # Update averages
            avg_pose_ms = alpha * pose_time + (1 - alpha) * avg_pose_ms if avg_pose_ms > 0 else pose_time

            # Calculate FPS
            frame_count += 1
            elapsed = time.time() - start_time
            fps_avg = frame_count / elapsed if elapsed > 0 else 0

            # Count occluded and border persons
            occluded_count = sum(1 for p in person_detections if p['occlusion']['is_occluded'])
            border_count = sum(1 for p in person_detections if p['border']['on_border'])
            too_close_count = sum(1 for p in person_detections if p['border'].get('too_close', False))
            standing_count = sum(1 for p in person_detections
                               if p['posture']['is_standing'] and p['posture']['confidence'] > 0.5)
            sitting_count = sum(1 for p in person_detections
                              if not p['posture']['is_standing'] and p['posture']['confidence'] > 0.5)

            # Draw HUD
            hud_y = 30
            cv2.putText(output, f"FPS: {fps_avg:.1f} | Inference: {pose_time:.0f}ms | Persons: {person_count}",
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
            elif occluded_count > 0:
                cv2.putText(output, f"Occluded: {occluded_count}/{person_count}",
                           (10, hud_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2, cv2.LINE_AA)
            else:
                cv2.putText(output, f"Standing: {standing_count} | Focal: {current_focal_length:.1f}px",
                           (10, hud_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

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
                            calibrated_fl = calibrate_focal_length(
                                person['bbox'], args.person_height,
                                args.calibrate_distance
                            )
                            if calibrated_fl is not None:
                                current_focal_length = calibrated_fl
                                print(f"\nFocal length calibrated: {current_focal_length:.1f}px")
                                print(f"  (Bbox height: {person['bbox_height']:.1f}px at {args.calibrate_distance}cm)")
                                break
                            else:
                                print("\nCalibration failed: invalid bbox")
                            break
                    else:
                        print("\nCalibration failed: no clear view of person (all occluded)")
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
        print(f"  Final focal length: {current_focal_length:.1f}px")
        print(f"{'='*70}\n")


def main():
    args = parse_args()

    print("Initializing YOLOv8-Pose model...")

    # Initialize YOLOv8-Pose
    pose_detector = YOLOv8PoseONNX(
        model_path=args.pose_model,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        input_size=(args.pose_size, args.pose_size)
    )

    print("\nModel loaded successfully!")
    print(f"Distance estimation based on bounding box height")
    print(f"Occlusion detection: {args.min_keypoints}/17 keypoints minimum")
    print(f"Person height: {args.person_height}cm\n")

    focal_length = args.focal_length
    if focal_length is not None:
        print(f"Using manual focal length: {focal_length}px")
    else:
        print("Focal length will be auto-estimated")

    # Run appropriate mode
    if args.image:
        run_image(args, pose_detector, focal_length)
    elif args.webcam:
        cap = cv2.VideoCapture(args.cam_id)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open webcam {args.cam_id}")
        run_video_stream(args, cap, f"Webcam {args.cam_id}", pose_detector, focal_length)
    elif args.video:
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {args.video}")
        run_video_stream(args, cap, args.video, pose_detector, focal_length)
    else:
        print("Error: Specify --image, --webcam, or --video")
        print("\nExamples:")
        print("  Image:   python run_yolov8poseDepth.py --image test.jpg")
        print("  Webcam:  python run_yolov8poseDepth.py --webcam")
        print("  Video:   python run_yolov8poseDepth.py --video input.mp4 --save-video output.mp4")
        print("  Calibrate: python run_yolov8poseDepth.py --webcam --calibrate --calibrate-distance 200")


if __name__ == "__main__":
    main()
