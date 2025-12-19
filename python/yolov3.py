import cv2
import numpy as np
import onnxruntime as ort


class YOLOv3ONNX:
    """YOLOv3 Object Detection Class using ONNX Runtime"""

    def __init__(self, model_path, conf_threshold=0.5, nms_threshold=0.4, input_size=(416, 416)):
        """
        Initialize YOLOv3 detector

        Args:
            model_path (str): Path to ONNX model file
            conf_threshold (float): Confidence threshold for filtering detections
            nms_threshold (float): Non-maximum suppression threshold
            input_size (tuple): Model input size (height, width)
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.input_size = input_size

        # Initialize ONNX Runtime session
        self.session = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

        # Get model input and output names
        self.input_names = [input.name for input in self.session.get_inputs()]
        self.output_names = [output.name for output in self.session.get_outputs()]

        # Print model info for debugging
        print(f"Model inputs: {self.input_names}")
        print(f"Model outputs: {self.output_names}")

        # COCO dataset class names
        self.class_names = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
            'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
            'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
            'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
            'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]

        # Generate random colors for each class
        np.random.seed(42)
        self.colors = np.random.randint(0, 255, size=(len(self.class_names), 3), dtype=np.uint8)

    def preprocess(self, image):
        """
        Preprocess image for YOLOv3 model

        Args:
            image (numpy.ndarray): Input image in BGR format

        Returns:
            tuple: Preprocessed image and scale factor
        """
        # Get original image dimensions
        height, width = image.shape[:2]

        # Resize image to model input size while maintaining aspect ratio
        scale = min(self.input_size[0] / height, self.input_size[1] / width)
        new_height, new_width = int(height * scale), int(width * scale)

        resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

        # Create canvas with model input size
        canvas = np.full((self.input_size[0], self.input_size[1], 3), 128, dtype=np.uint8)

        # Paste resized image on canvas (centered)
        y_offset = (self.input_size[0] - new_height) // 2
        x_offset = (self.input_size[1] - new_width) // 2
        canvas[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized

        # Convert to RGB and normalize to [0, 1]
        canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        canvas = canvas.astype(np.float32) / 255.0

        # Transpose to CHW format and add batch dimension
        canvas = np.transpose(canvas, (2, 0, 1))
        canvas = np.expand_dims(canvas, axis=0)

        return canvas, scale, x_offset, y_offset

    def postprocess(self, outputs, original_shape, scale, x_offset, y_offset):
        """
        Postprocess model outputs to get final detections

        Args:
            outputs: Model output (boxes, scores, num_detections)
            original_shape (tuple): Original image shape (height, width)
            scale (float): Scale factor used in preprocessing
            x_offset (int): X offset from preprocessing
            y_offset (int): Y offset from preprocessing

        Returns:
            list: List of detections [class_id, confidence, x1, y1, x2, y2]
        """
        detections = []

        # YOLOv3-12 ONNX model outputs:
        # Output 0: boxes [batch, num_boxes, 4] - format: [y1, x1, y2, x2] in normalized coordinates
        # Output 1: scores [batch, num_classes, num_boxes] - class probabilities
        # Output 2: num_detections [batch, 1] - number of valid detections

        boxes = outputs[0][0]  # Shape: [10647, 4]
        scores = outputs[1][0]  # Shape: [80, 10647]

        # Transpose scores to [num_boxes, num_classes]
        scores = scores.T  # Now shape: [10647, 80]

        # Debug: Check the max scores
        max_scores = np.max(scores, axis=1)
        top_10_indices = np.argsort(max_scores)[-10:][::-1]
        print(f"Top 10 confidence scores: {max_scores[top_10_indices]}")
        print(f"Top 10 class IDs: {np.argmax(scores[top_10_indices], axis=1)}")

        # Debug: Print box coordinates for top detections
        for idx in top_10_indices[:3]:
            print(f"Box {idx}: {boxes[idx]} (y1, x1, y2, x2)")

        # Process each box
        for i in range(len(boxes)):
            # Get the best class score for this box
            class_scores = scores[i]
            class_id = np.argmax(class_scores)
            confidence = class_scores[class_id]

            # Filter by confidence
            if confidence < self.conf_threshold:
                continue

            # Get box coordinates - these are already in absolute pixel coordinates
            # relative to the original image (before preprocessing)
            y1, x1, y2, x2 = boxes[i]

            # Debug first valid detection
            if len(detections) == 0:
                print(f"First valid detection: class={class_id}, conf={confidence:.3f}")
                print(f"  Raw box: y1={y1:.1f}, x1={x1:.1f}, y2={y2:.1f}, x2={x2:.1f}")
                print(f"  Image shape: {original_shape}")

            # Boxes are already in original image coordinates, just clip to boundaries
            x1_orig = max(0, min(x1, original_shape[1]))
            y1_orig = max(0, min(y1, original_shape[0]))
            x2_orig = max(0, min(x2, original_shape[1]))
            y2_orig = max(0, min(y2, original_shape[0]))

            # Skip invalid boxes
            if x2_orig <= x1_orig or y2_orig <= y1_orig:
                continue

            detections.append([int(class_id), float(confidence), x1_orig, y1_orig, x2_orig, y2_orig])

        print(f"Total detections after filtering: {len(detections)}")

        # Apply NMS to remove overlapping boxes
        if len(detections) > 0:
            detections = self.nms(detections)

        print(f"Detections after NMS: {len(detections)}")

        return detections

    def nms(self, detections):
        """
        Apply Non-Maximum Suppression to remove overlapping boxes

        Args:
            detections (list): List of detections

        Returns:
            list: Filtered detections after NMS
        """
        if len(detections) == 0:
            return []

        # Convert to numpy array for easier manipulation
        detections = np.array(detections)

        # Group detections by class
        classes = detections[:, 0].astype(int)
        unique_classes = np.unique(classes)

        final_detections = []

        for cls in unique_classes:
            # Get all detections for this class
            cls_mask = classes == cls
            cls_detections = detections[cls_mask]

            # Extract coordinates and confidences
            confidences = cls_detections[:, 1]
            boxes = cls_detections[:, 2:6]

            # Convert to [x, y, w, h] format for cv2.dnn.NMSBoxes
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
                self.nms_threshold
            )

            if len(indices) > 0:
                for idx in indices.flatten():
                    final_detections.append(cls_detections[idx].tolist())

        return final_detections

    def detect(self, image):
        """
        Detect objects in an image

        Args:
            image (numpy.ndarray): Input image in BGR format

        Returns:
            list: List of detections [class_id, confidence, x1, y1, x2, y2]
        """
        # Preprocess image
        input_tensor, scale, x_offset, y_offset = self.preprocess(image)

        # Prepare input feed dict
        # Handle different model input requirements
        input_feed = {}
        if len(self.input_names) == 1:
            # Single input model (just the image)
            input_feed[self.input_names[0]] = input_tensor
        else:
            # Multi-input model (image + image_shape)
            input_feed[self.input_names[0]] = input_tensor
            # image_shape should be [batch, 2] with original height and width
            image_shape = np.array([[image.shape[0], image.shape[1]]], dtype=np.float32)
            input_feed[self.input_names[1]] = image_shape

        # Run inference
        outputs = self.session.run(self.output_names, input_feed)

        # Postprocess outputs
        detections = self.postprocess(outputs, image.shape[:2], scale, x_offset, y_offset)

        return detections

    def draw_detections(self, image, detections):
        """
        Draw bounding boxes and labels on image

        Args:
            image (numpy.ndarray): Input image
            detections (list): List of detections

        Returns:
            numpy.ndarray: Image with drawn detections
        """
        output_image = image.copy()

        for detection in detections:
            class_id, confidence, x1, y1, x2, y2 = detection
            class_id = int(class_id)

            # Get class name and color
            class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"Class {class_id}"
            color = tuple(int(c) for c in self.colors[class_id])

            # Draw bounding box
            cv2.rectangle(output_image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

            # Draw label background
            label = f"{class_name}: {confidence:.2f}"
            (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(output_image, (int(x1), int(y1) - label_height - baseline - 5),
                         (int(x1) + label_width, int(y1)), color, -1)

            # Draw label text
            cv2.putText(output_image, label, (int(x1), int(y1) - baseline - 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return output_image

    def detect_and_draw(self, image):
        """
        Detect objects and draw results on image

        Args:
            image (numpy.ndarray): Input image in BGR format

        Returns:
            tuple: (detections, annotated_image)
        """
        detections = self.detect(image)
        annotated_image = self.draw_detections(image, detections)
        return detections, annotated_image


# Example usage
if __name__ == "__main__":
    # Initialize detector
    detector = YOLOv3ONNX(
        model_path="models/yolov3-12.onnx",
        conf_threshold=0.5,
        nms_threshold=0.4
    )

    # Load and detect on an image
    image = cv2.imread("test_image.jpg")
    if image is not None:
        detections, result = detector.detect_and_draw(image)

        # Print detections
        print(f"Found {len(detections)} objects:")
        for det in detections:
            class_id, conf, x1, y1, x2, y2 = det
            class_name = detector.class_names[int(class_id)]
            print(f"  - {class_name}: {conf:.2f} at [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")

        # Display result
        cv2.imshow("YOLOv3 Detection", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Error: Could not load image")
