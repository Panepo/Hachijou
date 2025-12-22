import cv2
import numpy as np
import onnxruntime as ort


class YOLOv8PoseONNX:
    """YOLOv8-Pose Keypoint Detection Class using ONNX Runtime"""

    def __init__(self, model_path, conf_threshold=0.5, iou_threshold=0.4, input_size=(640, 640)):
        """
        Initialize YOLOv8-Pose detector

        Args:
            model_path (str): Path to ONNX model file
            conf_threshold (float): Confidence threshold for filtering detections
            iou_threshold (float): IoU threshold for NMS
            input_size (tuple): Model input size (height, width)
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.input_size = input_size

        # Initialize ONNX Runtime session
        self.session = ort.InferenceSession(
            model_path,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )

        # Get model input and output names
        self.input_names = [input.name for input in self.session.get_inputs()]
        self.output_names = [output.name for output in self.session.get_outputs()]

        # Print model info
        print(f"[YOLOv8-Pose] Model loaded: {model_path}")
        print(f"[YOLOv8-Pose] Input names: {self.input_names}")
        print(f"[YOLOv8-Pose] Output names: {self.output_names}")
        print(f"[YOLOv8-Pose] Input size: {input_size}")

        # COCO 17 keypoints definition
        self.keypoint_names = [
            'nose',           # 0
            'left_eye',       # 1
            'right_eye',      # 2
            'left_ear',       # 3
            'right_ear',      # 4
            'left_shoulder',  # 5
            'right_shoulder', # 6
            'left_elbow',     # 7
            'right_elbow',    # 8
            'left_wrist',     # 9
            'right_wrist',    # 10
            'left_hip',       # 11
            'right_hip',      # 12
            'left_knee',      # 13
            'right_knee',     # 14
            'left_ankle',     # 15
            'right_ankle'     # 16
        ]

        # Skeleton connections for visualization (COCO format)
        self.skeleton = [
            [16, 14], [14, 12], [17, 15], [15, 13], [12, 13],  # legs
            [6, 12], [7, 13],  # torso
            [6, 8], [7, 9], [8, 10], [9, 11],  # arms
            [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]  # face to shoulders
        ]

        # Colors for keypoints and skeleton
        self.kpt_color = (0, 255, 0)  # Green for keypoints
        self.skeleton_color = (255, 100, 0)  # Blue for skeleton

        # Box color for person
        self.box_color = (255, 0, 255)  # Magenta

    def preprocess(self, image):
        """
        Preprocess image for YOLOv8-Pose model

        Args:
            image (numpy.ndarray): Input image in BGR format

        Returns:
            tuple: (preprocessed_tensor, scale, pad_w, pad_h)
        """
        height, width = image.shape[:2]

        # Calculate scale to fit input size while maintaining aspect ratio
        scale = min(self.input_size[0] / height, self.input_size[1] / width)
        new_height = int(height * scale)
        new_width = int(width * scale)

        # Resize image
        resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

        # Create padded canvas
        canvas = np.full((self.input_size[0], self.input_size[1], 3), 114, dtype=np.uint8)

        # Calculate padding to center the image
        pad_h = (self.input_size[0] - new_height) // 2
        pad_w = (self.input_size[1] - new_width) // 2

        canvas[pad_h:pad_h + new_height, pad_w:pad_w + new_width] = resized

        # Convert to RGB and normalize
        canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        canvas = canvas.astype(np.float32) / 255.0

        # Transpose to CHW format and add batch dimension
        canvas = np.transpose(canvas, (2, 0, 1))
        canvas = np.expand_dims(canvas, axis=0)

        return canvas, scale, pad_w, pad_h

    def postprocess(self, outputs, original_shape, scale, pad_w, pad_h):
        """
        Postprocess YOLOv8-Pose outputs

        YOLOv8-Pose output format:
        - Shape: [1, 56, 8400] for pose model
        - 56 channels = 4 (bbox) + 1 (obj_conf) + 51 (17 keypoints × 3: x, y, visibility)

        Args:
            outputs: Model output
            original_shape (tuple): Original image shape (height, width)
            scale (float): Scale factor used in preprocessing
            pad_w (int): Width padding
            pad_h (int): Height padding

        Returns:
            list: List of detections with keypoints
                  Each detection: {
                      'bbox': [x1, y1, x2, y2],
                      'confidence': float,
                      'keypoints': [[x, y, conf], ...] (17 keypoints)
                  }
        """
        predictions = outputs[0]  # Shape: [1, 56, 8400]
        predictions = predictions[0]  # Shape: [56, 8400]
        predictions = predictions.T  # Shape: [8400, 56]

        detections = []
        orig_height, orig_width = original_shape

        for pred in predictions:
            # Extract bbox (first 4 values: x_center, y_center, width, height)
            x_center, y_center, w, h = pred[0:4]

            # Extract confidence (5th value)
            confidence = pred[4]

            # Filter by confidence
            if confidence < self.conf_threshold:
                continue

            # Convert from xywh to xyxy format
            x1 = x_center - w / 2
            y1 = y_center - h / 2
            x2 = x_center + w / 2
            y2 = y_center + h / 2

            # Rescale to original image coordinates
            x1 = (x1 - pad_w) / scale
            y1 = (y1 - pad_h) / scale
            x2 = (x2 - pad_w) / scale
            y2 = (y2 - pad_h) / scale

            # Clip to image boundaries
            x1 = max(0, min(x1, orig_width))
            y1 = max(0, min(y1, orig_height))
            x2 = max(0, min(x2, orig_width))
            y2 = max(0, min(y2, orig_height))

            # Skip invalid boxes
            if x2 <= x1 or y2 <= y1:
                continue

            # Extract keypoints (remaining 51 values: 17 keypoints × 3)
            keypoints = []
            for i in range(17):
                kpt_x = pred[5 + i * 3]
                kpt_y = pred[5 + i * 3 + 1]
                kpt_conf = pred[5 + i * 3 + 2]

                # Rescale keypoints to original image
                kpt_x = (kpt_x - pad_w) / scale
                kpt_y = (kpt_y - pad_h) / scale

                keypoints.append([kpt_x, kpt_y, kpt_conf])

            detections.append({
                'bbox': [x1, y1, x2, y2],
                'confidence': float(confidence),
                'keypoints': keypoints
            })

        # Apply NMS
        if len(detections) > 0:
            detections = self.nms(detections)

        return detections

    def nms(self, detections):
        """
        Apply Non-Maximum Suppression

        Args:
            detections (list): List of detections

        Returns:
            list: Filtered detections
        """
        if len(detections) == 0:
            return []

        # Extract boxes and confidences
        boxes = np.array([det['bbox'] for det in detections])
        confidences = np.array([det['confidence'] for det in detections])

        # Convert [x1, y1, x2, y2] to [x, y, w, h]
        boxes_xywh = np.zeros_like(boxes)
        boxes_xywh[:, 0] = boxes[:, 0]
        boxes_xywh[:, 1] = boxes[:, 1]
        boxes_xywh[:, 2] = boxes[:, 2] - boxes[:, 0]
        boxes_xywh[:, 3] = boxes[:, 3] - boxes[:, 1]

        # Apply NMS
        indices = cv2.dnn.NMSBoxes(
            boxes_xywh.tolist(),
            confidences.tolist(),
            self.conf_threshold,
            self.iou_threshold
        )

        if len(indices) > 0:
            return [detections[i] for i in indices.flatten()]
        return []

    def detect(self, image):
        """
        Detect poses in an image

        Args:
            image (numpy.ndarray): Input image in BGR format

        Returns:
            list: List of detections with keypoints
        """
        # Preprocess
        input_tensor, scale, pad_w, pad_h = self.preprocess(image)

        # Run inference
        outputs = self.session.run(self.output_names, {self.input_names[0]: input_tensor})

        # Postprocess
        detections = self.postprocess(outputs, image.shape[:2], scale, pad_w, pad_h)

        return detections

    def draw_detections(self, image, detections, draw_bbox=True, draw_keypoints=True,
                       draw_skeleton=True, kpt_threshold=0.5):
        """
        Draw detections on image

        Args:
            image (numpy.ndarray): Input image
            detections (list): List of detections
            draw_bbox (bool): Whether to draw bounding boxes
            draw_keypoints (bool): Whether to draw keypoints
            draw_skeleton (bool): Whether to draw skeleton
            kpt_threshold (float): Confidence threshold for drawing keypoints

        Returns:
            numpy.ndarray: Image with drawn detections
        """
        output_image = image.copy()

        for det in detections:
            bbox = det['bbox']
            confidence = det['confidence']
            keypoints = det['keypoints']

            x1, y1, x2, y2 = map(int, bbox)

            # Draw bounding box
            if draw_bbox:
                cv2.rectangle(output_image, (x1, y1), (x2, y2), self.box_color, 2)

                # Draw label
                label = f"Person: {confidence:.2f}"
                (label_w, label_h), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
                )
                cv2.rectangle(output_image, (x1, y1 - label_h - baseline - 5),
                            (x1 + label_w, y1), self.box_color, -1)
                cv2.putText(output_image, label, (x1, y1 - baseline - 2),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            # Draw skeleton first (so keypoints are on top)
            if draw_skeleton:
                for sk in self.skeleton:
                    kpt1_idx = sk[0] - 1  # Convert to 0-indexed
                    kpt2_idx = sk[1] - 1

                    if kpt1_idx >= len(keypoints) or kpt2_idx >= len(keypoints):
                        continue

                    kpt1 = keypoints[kpt1_idx]
                    kpt2 = keypoints[kpt2_idx]

                    # Only draw if both keypoints are confident
                    if kpt1[2] > kpt_threshold and kpt2[2] > kpt_threshold:
                        pt1 = (int(kpt1[0]), int(kpt1[1]))
                        pt2 = (int(kpt2[0]), int(kpt2[1]))
                        cv2.line(output_image, pt1, pt2, self.skeleton_color, 2)

            # Draw keypoints
            if draw_keypoints:
                for i, (kpt_x, kpt_y, kpt_conf) in enumerate(keypoints):
                    if kpt_conf > kpt_threshold:
                        pt = (int(kpt_x), int(kpt_y))
                        cv2.circle(output_image, pt, 4, self.kpt_color, -1)
                        cv2.circle(output_image, pt, 5, (0, 0, 0), 1)  # Black outline

        return output_image

    def detect_and_draw(self, image, **draw_kwargs):
        """
        Detect poses and draw results on image

        Args:
            image (numpy.ndarray): Input image in BGR format
            **draw_kwargs: Additional arguments for draw_detections

        Returns:
            tuple: (detections, annotated_image)
        """
        detections = self.detect(image)
        annotated_image = self.draw_detections(image, detections, **draw_kwargs)
        return detections, annotated_image


if __name__ == "__main__":
    # Example usage
    detector = YOLOv8PoseONNX(
        model_path="models/yolov8m-pose.onnx",
        conf_threshold=0.5,
        iou_threshold=0.4
    )

    # Test on image
    image = cv2.imread("test_image.jpg")
    if image is not None:
        detections, result = detector.detect_and_draw(image)

        print(f"Found {len(detections)} persons:")
        for i, det in enumerate(detections):
            print(f"\nPerson {i+1}:")
            print(f"  Confidence: {det['confidence']:.3f}")
            print(f"  Bounding box: {det['bbox']}")
            print(f"  Visible keypoints:")
            for j, (x, y, conf) in enumerate(det['keypoints']):
                if conf > 0.5:
                    print(f"    {detector.keypoint_names[j]}: ({x:.1f}, {y:.1f}) [{conf:.3f}]")

        cv2.imshow("YOLOv8-Pose", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Error: Could not load image")
