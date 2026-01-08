import cv2
import numpy as np
import onnxruntime as ort


class YOLOv11SegONNX:
    """YOLOv11 Segmentation Class using ONNX Runtime"""

    def __init__(self, model_path, conf_threshold=0.5, nms_threshold=0.4, input_size=(640, 640)):
        """
        Initialize YOLOv11 Segmentation detector

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
        for i, output in enumerate(self.session.get_outputs()):
            print(f"  Output {i} ({output.name}): {output.shape}")

        # COCO dataset class names (80 classes)
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
        Preprocess image for YOLOv11 model

        Args:
            image (numpy.ndarray): Input image in BGR format

        Returns:
            tuple: Preprocessed image, scale factor, padding info
        """
        # Get original image dimensions
        height, width = image.shape[:2]

        # Resize image to model input size while maintaining aspect ratio
        scale = min(self.input_size[0] / height, self.input_size[1] / width)
        new_height, new_width = int(height * scale), int(width * scale)

        resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

        # Create canvas with model input size (letterbox)
        canvas = np.full((self.input_size[0], self.input_size[1], 3), 114, dtype=np.uint8)

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
        Postprocess model outputs to get final detections and masks

        Args:
            outputs: Model outputs (detections and proto masks)
            original_shape (tuple): Original image shape (height, width)
            scale (float): Scale factor used in preprocessing
            x_offset (int): X offset from preprocessing
            y_offset (int): Y offset from preprocessing

        Returns:
            tuple: (detections list, masks list)
                   detections: [class_id, confidence, x1, y1, x2, y2]
                   masks: corresponding segmentation masks
        """
        detections = []
        mask_coefficients = []

        # YOLOv11-seg ONNX model outputs:
        # Output 0: [1, 116, 8400] - detections (4 bbox + 80 classes + 32 mask coefficients)
        # Output 1: [1, 32, 160, 160] - prototype masks

        predictions = outputs[0][0]  # Shape: [116, 8400]
        proto_masks = outputs[1][0]  # Shape: [32, 160, 160]

        # Transpose predictions to [8400, 116]
        predictions = predictions.T

        # Split predictions: [cx, cy, w, h, class_scores..., mask_coeffs...]
        boxes = predictions[:, :4]  # [8400, 4] - center format
        class_scores = predictions[:, 4:84]  # [8400, 80]
        mask_coeffs = predictions[:, 84:]  # [8400, 32]

        # Get max class score and class id for each detection
        max_scores = np.max(class_scores, axis=1)
        class_ids = np.argmax(class_scores, axis=1)

        # Filter by confidence threshold
        valid_mask = max_scores > self.conf_threshold
        valid_boxes = boxes[valid_mask]
        valid_scores = max_scores[valid_mask]
        valid_class_ids = class_ids[valid_mask]
        valid_mask_coeffs = mask_coeffs[valid_mask]

        if len(valid_boxes) == 0:
            return [], []

        # Convert boxes from center format (cx, cy, w, h) to corner format (x1, y1, x2, y2)
        # and scale back to original image coordinates
        x_center = valid_boxes[:, 0]
        y_center = valid_boxes[:, 1]
        w = valid_boxes[:, 2]
        h = valid_boxes[:, 3]

        # Convert to corner format relative to input image
        x1 = x_center - w / 2
        y1 = y_center - h / 2
        x2 = x_center + w / 2
        y2 = y_center + h / 2

        # Remove padding offset
        x1 = (x1 - x_offset)
        y1 = (y1 - y_offset)
        x2 = (x2 - x_offset)
        y2 = (y2 - y_offset)

        # Scale back to original image size
        x1 = x1 / scale
        y1 = y1 / scale
        x2 = x2 / scale
        y2 = y2 / scale

        # Clip to image boundaries
        x1 = np.clip(x1, 0, original_shape[1])
        y1 = np.clip(y1, 0, original_shape[0])
        x2 = np.clip(x2, 0, original_shape[1])
        y2 = np.clip(y2, 0, original_shape[0])

        # Apply NMS
        boxes_for_nms = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1)
        indices = cv2.dnn.NMSBoxes(
            boxes_for_nms.tolist(),
            valid_scores.tolist(),
            self.conf_threshold,
            self.nms_threshold
        )

        if len(indices) > 0:
            for idx in indices.flatten():
                detections.append([
                    int(valid_class_ids[idx]),
                    float(valid_scores[idx]),
                    float(x1[idx]),
                    float(y1[idx]),
                    float(x2[idx]),
                    float(y2[idx])
                ])
                mask_coefficients.append(valid_mask_coeffs[idx])

        # Generate masks for detections
        masks = []
        if len(mask_coefficients) > 0:
            masks = self.process_masks(
                np.array(mask_coefficients),
                proto_masks,
                detections,
                original_shape,
                scale,
                x_offset,
                y_offset
            )

        return detections, masks

    def process_masks(self, mask_coeffs, proto_masks, detections, original_shape, scale, x_offset, y_offset):
        """
        Generate instance segmentation masks

        Args:
            mask_coeffs: Mask coefficients [N, 32]
            proto_masks: Prototype masks [32, 160, 160]
            detections: List of detections
            original_shape: Original image shape
            scale: Scale factor
            x_offset: X offset
            y_offset: Y offset

        Returns:
            List of binary masks for each detection
        """
        masks = []
        proto_h, proto_w = proto_masks.shape[1:]

        # Generate masks by matrix multiplication
        # mask_coeffs: [N, 32], proto_masks: [32, 160, 160] -> masks: [N, 160, 160]
        mask_protos = np.matmul(mask_coeffs, proto_masks.reshape(32, -1))
        mask_protos = mask_protos.reshape(-1, proto_h, proto_w)

        # Apply sigmoid activation
        mask_protos = 1 / (1 + np.exp(-mask_protos))

        for i, detection in enumerate(detections):
            _, _, x1, y1, x2, y2 = detection

            # Get mask for this detection
            mask = mask_protos[i]

            # Resize mask to input size
            mask_resized = cv2.resize(mask, self.input_size, interpolation=cv2.INTER_LINEAR)

            # Remove padding
            mask_cropped = mask_resized[y_offset:y_offset + int(original_shape[0] * scale),
                                       x_offset:x_offset + int(original_shape[1] * scale)]

            # Resize to original image size
            mask_full = cv2.resize(mask_cropped, (original_shape[1], original_shape[0]),
                                  interpolation=cv2.INTER_LINEAR)

            # Crop mask to bounding box
            mask_bbox = np.zeros_like(mask_full)
            x1_int, y1_int = int(x1), int(y1)
            x2_int, y2_int = int(x2), int(y2)
            mask_bbox[y1_int:y2_int, x1_int:x2_int] = mask_full[y1_int:y2_int, x1_int:x2_int]

            # Threshold mask
            mask_binary = (mask_bbox > 0.5).astype(np.uint8)
            masks.append(mask_binary)

        return masks

    def detect(self, image):
        """
        Detect objects and generate segmentation masks in an image

        Args:
            image (numpy.ndarray): Input image in BGR format

        Returns:
            tuple: (detections, masks)
                   detections: List of [class_id, confidence, x1, y1, x2, y2]
                   masks: List of binary masks
        """
        # Preprocess image
        input_tensor, scale, x_offset, y_offset = self.preprocess(image)

        # Run inference
        outputs = self.session.run(self.output_names, {self.input_names[0]: input_tensor})

        # Postprocess outputs
        detections, masks = self.postprocess(outputs, image.shape[:2], scale, x_offset, y_offset)

        return detections, masks

    def draw_detections(self, image, detections, masks=None, show_masks=True):
        """
        Draw bounding boxes, labels, and segmentation masks on image

        Args:
            image (numpy.ndarray): Input image
            detections (list): List of detections
            masks (list): List of segmentation masks
            show_masks (bool): Whether to overlay segmentation masks

        Returns:
            numpy.ndarray: Image with drawn detections
        """
        output_image = image.copy()

        # Draw masks first (as overlay)
        if show_masks and masks is not None and len(masks) > 0:
            mask_overlay = np.zeros_like(image, dtype=np.uint8)

            for i, (detection, mask) in enumerate(zip(detections, masks)):
                class_id = int(detection[0])
                color = tuple(int(c) for c in self.colors[class_id])

                # Apply colored mask
                for c in range(3):
                    mask_overlay[:, :, c] = np.where(mask == 1,
                                                     mask_overlay[:, :, c] + color[c] // 2,
                                                     mask_overlay[:, :, c])

            # Blend mask overlay with original image
            output_image = cv2.addWeighted(output_image, 1.0, mask_overlay, 0.5, 0)

        # Draw bounding boxes and labels
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

    def detect_and_draw(self, image, show_masks=True):
        """
        Detect objects, generate masks, and draw results on image

        Args:
            image (numpy.ndarray): Input image in BGR format
            show_masks (bool): Whether to overlay segmentation masks

        Returns:
            tuple: (detections, masks, annotated_image)
        """
        detections, masks = self.detect(image)
        annotated_image = self.draw_detections(image, detections, masks, show_masks)
        return detections, masks, annotated_image


# Example usage
if __name__ == "__main__":
    # Initialize detector
    detector = YOLOv11SegONNX(
        model_path="models/yolov11m-seg.onnx",
        conf_threshold=0.5,
        nms_threshold=0.4
    )

    # Load and detect on an image
    image = cv2.imread("test_image.jpg")
    if image is not None:
        detections, masks, result = detector.detect_and_draw(image)

        # Print detections
        print(f"Found {len(detections)} objects:")
        for i, det in enumerate(detections):
            class_id, conf, x1, y1, x2, y2 = det
            class_name = detector.class_names[int(class_id)]
            print(f"  {i+1}. {class_name}: {conf:.2f} at [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")

        # Display result
        cv2.imshow("YOLOv11 Segmentation", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Error: Could not load image")
