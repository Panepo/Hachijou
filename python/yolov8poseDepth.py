# yolov8poseDepth.py
import cv2
import numpy as np
import math
from yolov8pose import YOLOv8PoseONNX


class YOLOv8PoseDepth:
    """
    YOLOv8-Pose with Distance Estimation
    Extends YOLOv8-Pose with distance calculation, occlusion detection, and posture analysis
    """

    def __init__(self, pose_model_path, person_height_cm=180.0, focal_length=None,
                 conf_threshold=0.5, iou_threshold=0.4, input_size=640,
                 min_keypoints=10, border_threshold=10, border_distance_factor=0.6,
                 kpt_threshold=0.5):
        """
        Initialize YOLOv8-Pose with depth estimation

        Args:
            pose_model_path: Path to YOLOv8-Pose ONNX model
            person_height_cm: Assumed person height in cm (default: 180.0cm)
            focal_length: Camera focal length in pixels (auto-calibrate if None)
            conf_threshold: Pose detection confidence threshold
            iou_threshold: IoU threshold for NMS
            input_size: Model input size
            min_keypoints: Minimum visible keypoints for non-occluded person
            border_threshold: Pixel threshold for border detection
            border_distance_factor: Distance multiplier for border detections
            kpt_threshold: Keypoint confidence threshold
        """
        # Initialize pose detector
        self.pose_detector = YOLOv8PoseONNX(
            model_path=pose_model_path,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            input_size=(input_size, input_size)
        )

        # Distance estimation parameters
        self.person_height_cm = person_height_cm
        self.focal_length = focal_length
        self.min_keypoints = min_keypoints
        self.border_threshold = border_threshold
        self.border_distance_factor = border_distance_factor
        self.kpt_threshold = kpt_threshold

    def estimate_focal_length(self, image_width, image_height, fov_horizontal=60.0):
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

    def get_keypoint_span_height(self, keypoints):
        """
        Calculate the vertical span of visible keypoints

        Args:
            keypoints: List of keypoints [[x, y, conf], ...]

        Returns:
            float: Vertical span in pixels (max_y - min_y)
        """
        visible_y = [kpt[1] for kpt in keypoints if kpt[2] > self.kpt_threshold]

        if len(visible_y) < 2:
            return 0.0

        return max(visible_y) - min(visible_y)

    def is_bbox_on_border(self, bbox, frame_shape):
        """
        Detect if bounding box touches or is near the frame border

        Args:
            bbox: [x1, y1, x2, y2]
            frame_shape: (height, width) of the frame

        Returns:
            dict: {'on_border': bool, 'edges': list of edge names}
        """
        x1, y1, x2, y2 = bbox
        frame_height, frame_width = frame_shape[:2]

        edges = []

        if x1 <= self.border_threshold:
            edges.append('left')
        if x2 >= frame_width - self.border_threshold:
            edges.append('right')
        if y1 <= self.border_threshold:
            edges.append('top')
        if y2 >= frame_height - self.border_threshold:
            edges.append('bottom')

        return {
            'on_border': len(edges) > 0,
            'edges': edges
        }

    def detect_standing_posture(self, keypoints):
        """
        Detect if person is in standing posture based on body keypoint alignment

        Args:
            keypoints: List of 17 keypoints [[x, y, conf], ...]

        Returns:
            dict: Standing posture analysis
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
        shoulder_visible = (left_shoulder[2] > self.kpt_threshold or right_shoulder[2] > self.kpt_threshold)
        hip_visible = (left_hip[2] > self.kpt_threshold or right_hip[2] > self.kpt_threshold)
        knee_visible = (left_knee[2] > self.kpt_threshold or right_knee[2] > self.kpt_threshold)
        ankle_visible = (left_ankle[2] > self.kpt_threshold or right_ankle[2] > self.kpt_threshold)

        # Can't determine if critical points are missing
        if not (shoulder_visible and hip_visible):
            return {
                'is_standing': False,
                'confidence': 0.0,
                'reason': 'insufficient_keypoints',
                'vertical_alignment': 0.0
            }

        # Calculate average positions for each body part
        shoulder_y = (left_shoulder[1] if left_shoulder[2] > self.kpt_threshold else 0) + \
                     (right_shoulder[1] if right_shoulder[2] > self.kpt_threshold else 0)
        shoulder_count = (1 if left_shoulder[2] > self.kpt_threshold else 0) + \
                         (1 if right_shoulder[2] > self.kpt_threshold else 0)
        shoulder_y /= max(shoulder_count, 1)

        hip_y = (left_hip[1] if left_hip[2] > self.kpt_threshold else 0) + \
                (right_hip[1] if right_hip[2] > self.kpt_threshold else 0)
        hip_count = (1 if left_hip[2] > self.kpt_threshold else 0) + \
                    (1 if right_hip[2] > self.kpt_threshold else 0)
        hip_y /= max(hip_count, 1)

        knee_y = (left_knee[1] if left_knee[2] > self.kpt_threshold else 0) + \
                 (right_knee[1] if right_knee[2] > self.kpt_threshold else 0)
        knee_count = (1 if left_knee[2] > self.kpt_threshold else 0) + \
                     (1 if right_knee[2] > self.kpt_threshold else 0)
        knee_y /= max(knee_count, 1) if knee_count > 0 else 1

        ankle_y = (left_ankle[1] if left_ankle[2] > self.kpt_threshold else 0) + \
                  (right_ankle[1] if right_ankle[2] > self.kpt_threshold else 0)
        ankle_count = (1 if left_ankle[2] > self.kpt_threshold else 0) + \
                      (1 if right_ankle[2] > self.kpt_threshold else 0)
        ankle_y /= max(ankle_count, 1) if ankle_count > 0 else 1

        # Calculate torso height (shoulder to hip)
        torso_height = abs(hip_y - shoulder_y)

        # Calculate vertical alignment metrics
        vertical_alignment = 0.0

        if knee_visible and ankle_visible and torso_height > 0:
            # Full body visible
            hip_knee_height = abs(knee_y - hip_y)
            knee_ankle_height = abs(ankle_y - knee_y)

            shoulder_to_hip_ok = hip_y > shoulder_y
            hip_to_knee_ok = knee_y > hip_y
            knee_to_ankle_ok = ankle_y > knee_y

            total_height = ankle_y - shoulder_y

            if total_height > 0:
                torso_ratio = torso_height / total_height
                leg_ratio = (hip_knee_height + knee_ankle_height) / total_height

                is_proportional = (0.30 < torso_ratio < 0.55) and (0.45 < leg_ratio < 0.70)

                alignment_checks = [shoulder_to_hip_ok, hip_to_knee_ok, knee_to_ankle_ok, is_proportional]
                vertical_alignment = sum(alignment_checks) / len(alignment_checks)

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

            upper_body_height = knee_y - shoulder_y
            if upper_body_height > 0:
                torso_ratio = torso_height / upper_body_height

                is_proportional = 0.35 < torso_ratio < 0.65

                alignment_checks = [shoulder_to_hip_ok, hip_to_knee_ok, is_proportional]
                vertical_alignment = sum(alignment_checks) / len(alignment_checks)

                if vertical_alignment >= 0.66:
                    return {
                        'is_standing': True,
                        'confidence': vertical_alignment * 0.8,
                        'reason': 'vertical_upper_body',
                        'vertical_alignment': vertical_alignment
                    }

        return {
            'is_standing': False,
            'confidence': 0.6,
            'reason': 'unclear_posture',
            'vertical_alignment': vertical_alignment
        }

    def detect_occlusion(self, keypoints):
        """
        Detect if person is occluded based on missing keypoints

        Args:
            keypoints: List of 17 keypoints [[x, y, conf], ...]

        Returns:
            dict: Occlusion analysis
        """
        # Define keypoint groups
        upper_body = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        lower_body = [11, 12, 13, 14, 15, 16]

        visible_keypoints = []
        missing_keypoints = []
        visible_upper = 0
        visible_lower = 0

        for i, (x, y, conf) in enumerate(keypoints):
            if conf > self.kpt_threshold:
                visible_keypoints.append(i)
                if i in upper_body:
                    visible_upper += 1
                if i in lower_body:
                    visible_lower += 1
            else:
                missing_keypoints.append(i)

        num_visible = len(visible_keypoints)
        is_occluded = False
        occlusion_type = 'none'

        if num_visible < 10:
            is_occluded = True
            occlusion_type = 'partial'
        elif visible_lower < 4 and visible_upper >= 3:
            is_occluded = True
            occlusion_type = 'lower_body'
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

    def calculate_distance_from_bbox(self, bbox, occlusion_info=None, posture_info=None):
        """
        Calculate distance using bounding box height
        Distance = (Real Height × Focal Length) / Pixel Height

        Args:
            bbox: Bounding box [x1, y1, x2, y2]
            occlusion_info: Occlusion detection dict (optional)
            posture_info: Posture detection dict (optional)

        Returns:
            tuple: (distance_cm, effective_height, height_source) or (None, person_height_cm, "unknown")
        """
        if self.focal_length is None or self.focal_length <= 0:
            return None, self.person_height_cm, "unknown"

        x1, y1, x2, y2 = bbox
        pixel_height = y2 - y1

        if pixel_height <= 0:
            return None, self.person_height_cm, "unknown"

        # Adjust person height based on posture and occlusion
        effective_height = self.person_height_cm
        height_source = "full_standing"

        # Check posture first
        if posture_info is not None and posture_info['confidence'] > 0.5:
            if not posture_info['is_standing']:
                effective_height = self.person_height_cm * 0.60
                height_source = "sitting_fixed"

        # Fall back to occlusion heuristics
        elif occlusion_info is not None:
            bbox_width = x2 - x1
            bbox_height = y2 - y1
            aspect_ratio = bbox_width / bbox_height if bbox_height > 0 else 0
            is_near_square = 0.8 <= aspect_ratio <= 1.2

            if is_near_square and occlusion_info['occlusion_type'] == 'partial':
                effective_height = self.person_height_cm / 3
                height_source = "crouched"
            elif occlusion_info['occlusion_type'] == 'partial':
                effective_height = self.person_height_cm / 2.5
                height_source = "partial_occluded"
            elif occlusion_info['occlusion_type'] == 'lower_body':
                effective_height = self.person_height_cm / 2.0
                height_source = "upper_body_only"
            elif occlusion_info['occlusion_type'] == 'upper_body':
                effective_height = self.person_height_cm / 2.0
                height_source = "lower_body_only"

        distance = (effective_height * self.focal_length) / pixel_height
        return distance, effective_height, height_source

    def calibrate_focal_length(self, bbox, known_distance_cm):
        """
        Calibrate focal length based on known distance

        Args:
            bbox: Bounding box [x1, y1, x2, y2]
            known_distance_cm: Known distance to person in cm

        Returns:
            float: Calibrated focal length in pixels, or None
        """
        x1, y1, x2, y2 = bbox
        pixel_height = y2 - y1

        if pixel_height <= 0:
            return None

        focal_length = (pixel_height * known_distance_cm) / self.person_height_cm
        return focal_length

    def draw_person_with_info(self, image, detection, distance_cm, effective_height, height_source,
                              occlusion_info, posture_info, border_info,
                              draw_keypoints=True, draw_skeleton=True):
        """
        Draw person with pose, distance, posture, and occlusion information

        Args:
            image: Image to draw on
            detection: Detection dict from YOLOv8-Pose
            distance_cm: Calculated distance in cm
            effective_height: Effective height used for calculation
            height_source: Source of height estimation
            occlusion_info: Occlusion analysis dict
            posture_info: Posture detection dict
            border_info: Border detection dict
            draw_keypoints: Whether to draw keypoints
            draw_skeleton: Whether to draw skeleton

        Returns:
            numpy.ndarray: Image with annotations
        """
        bbox = detection['bbox']
        confidence = detection['confidence']
        keypoints = detection['keypoints']

        x1, y1, x2, y2 = map(int, bbox)

        # Choose color based on proximity and occlusion
        if border_info.get('too_close', False):
            box_color = (0, 0, 255)  # Red
        elif border_info['on_border']:
            box_color = (0, 100, 255)  # Orange-red
        elif occlusion_info['is_occluded']:
            box_color = (0, 165, 255)  # Orange
        elif posture_info and not posture_info['is_standing'] and posture_info['confidence'] > 0.5:
            box_color = (255, 165, 0)  # Blue
        else:
            box_color = (0, 255, 0)  # Green

        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), box_color, 2)

        # Draw skeleton
        if draw_skeleton:
            for sk in self.pose_detector.skeleton:
                kpt1_idx = sk[0] - 1
                kpt2_idx = sk[1] - 1

                if kpt1_idx >= len(keypoints) or kpt2_idx >= len(keypoints):
                    continue

                kpt1 = keypoints[kpt1_idx]
                kpt2 = keypoints[kpt2_idx]

                if kpt1[2] > self.kpt_threshold and kpt2[2] > self.kpt_threshold:
                    pt1 = (int(kpt1[0]), int(kpt1[1]))
                    pt2 = (int(kpt2[0]), int(kpt2[1]))
                    cv2.line(image, pt1, pt2, (255, 150, 0), 2)

        # Draw keypoints
        if draw_keypoints:
            for i, (kpt_x, kpt_y, kpt_conf) in enumerate(keypoints):
                if kpt_conf > self.kpt_threshold:
                    pt = (int(kpt_x), int(kpt_y))
                    cv2.circle(image, pt, 4, (0, 255, 0), -1)
                    cv2.circle(image, pt, 5, (0, 0, 0), 1)
                elif kpt_conf > 0.1:
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

    def process_frame(self, frame, draw_keypoints=True, draw_skeleton=True):
        """
        Process a single frame with pose detection and distance estimation

        Args:
            frame: Input frame
            draw_keypoints: Whether to draw keypoints
            draw_skeleton: Whether to draw skeleton

        Returns:
            tuple: (output_image, num_persons, detections_info)
        """
        # Run pose detection
        detections = self.pose_detector.detect(frame)

        # Create output image
        output = frame.copy()
        person_detections = []

        # Process each detected person
        for detection in detections:
            bbox = detection['bbox']
            keypoints = detection['keypoints']

            # Border detection
            border_info = self.is_bbox_on_border(bbox, frame.shape)

            # Occlusion detection
            occlusion_info = self.detect_occlusion(keypoints)

            # Posture detection
            posture_info = self.detect_standing_posture(keypoints)

            # Distance calculation
            result = self.calculate_distance_from_bbox(bbox, occlusion_info, posture_info)

            if result is not None and result[0] is not None:
                distance_cm, effective_height, height_source = result
            else:
                distance_cm, effective_height, height_source = None, self.person_height_cm, "unknown"

            # Adjust distance for border cases
            adjusted_distance_cm = distance_cm
            is_too_close = False

            if distance_cm is not None:
                if border_info['on_border'] and occlusion_info['visible_keypoints'] < self.min_keypoints:
                    is_too_close = True
                    adjusted_distance_cm = distance_cm * self.border_distance_factor

            border_info['too_close'] = is_too_close

            # Draw annotations
            output = self.draw_person_with_info(
                output, detection, adjusted_distance_cm, effective_height, height_source,
                occlusion_info, posture_info, border_info,
                draw_keypoints, draw_skeleton
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

        return output, len(person_detections), person_detections
