import cv2
import numpy as np
import onnxruntime as ort


class Mask2FormerONNX:
    """Mask2Former Segmentation Class using ONNX Runtime"""

    def __init__(self, model_path, conf_threshold=0.5, input_size=(384, 384)):
        """
        Initialize Mask2Former segmentation model

        Args:
            model_path (str): Path to ONNX model file
            conf_threshold (float): Confidence threshold for filtering detections
            input_size (tuple): Model input size (height, width)
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
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

        # COCO-Stuff 134 classes (COCO 80 things + 54 stuff classes)
        self.class_names = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
            'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
            'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
            'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
            'scissors', 'teddy bear', 'hair drier', 'toothbrush',
            # Stuff classes (80-133)
            'banner', 'blanket', 'bridge', 'cardboard', 'counter', 'curtain', 'door-stuff', 'floor-wood',
            'flower', 'fruit', 'gravel', 'house', 'light', 'mirror-stuff', 'net', 'pillow', 'platform',
            'playingfield', 'railroad', 'river', 'road', 'roof', 'sand', 'sea', 'shelf', 'snow', 'stairs',
            'tent', 'towel', 'wall-brick', 'wall-stone', 'wall-tile', 'wall-wood', 'water-other', 'window-blind',
            'window-other', 'tree-merged', 'fence-merged', 'ceiling-merged', 'sky-other-merged', 'cabinet-merged',
            'table-merged', 'floor-other-merged', 'pavement-merged', 'mountain-merged', 'grass-merged', 'dirt-merged',
            'paper-merged', 'food-other-merged', 'building-other-merged', 'rock-merged', 'wall-other-merged', 'rug-merged'
        ]

        # Generate random colors for each class
        np.random.seed(42)
        self.colors = np.random.randint(0, 255, size=(len(self.class_names), 3), dtype=np.uint8)

    def preprocess(self, image):
        """
        Preprocess image for Mask2Former model

        Args:
            image (numpy.ndarray): Input image in BGR format

        Returns:
            tuple: Preprocessed image and metadata
        """
        # Store original dimensions
        original_height, original_width = image.shape[:2]

        # Resize image to model input size
        resized = cv2.resize(image, (self.input_size[1], self.input_size[0]), interpolation=cv2.INTER_LINEAR)

        # Convert to RGB
        rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # Normalize using ImageNet stats
        mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
        std = np.array([58.395, 57.12, 57.375], dtype=np.float32)

        normalized = (rgb_image.astype(np.float32) - mean) / std

        # Transpose to CHW format and add batch dimension
        input_tensor = np.transpose(normalized, (2, 0, 1))
        input_tensor = np.expand_dims(input_tensor, axis=0)

        return input_tensor, original_height, original_width

    def postprocess(self, outputs, original_height, original_width):
        """
        Postprocess model outputs to get segmentation masks

        Args:
            outputs: Model outputs (class_idx, masks)
            original_height (int): Original image height
            original_width (int): Original image width

        Returns:
            numpy.ndarray: Segmentation map with class IDs
        """
        # Mask2Former outputs: [class_idx (1, 100, 134), masks (1, 100, 96, 96)]
        class_logits = outputs[0][0]  # (100, 134) - 100 instances, 134 classes
        mask_logits = outputs[1][0]   # (100, 96, 96) - 100 instance masks

        # Get predicted class for each instance
        pred_classes = np.argmax(class_logits, axis=1)  # (100,)

        # Get confidence scores using softmax
        class_probs = self._softmax(class_logits, axis=1)
        pred_scores = np.max(class_probs, axis=1)  # (100,)

        # Apply sigmoid to mask logits to get probabilities
        mask_probs = 1 / (1 + np.exp(-mask_logits))  # (100, 96, 96)

        # Create segmentation map
        seg_map = np.zeros((mask_logits.shape[1], mask_logits.shape[2]), dtype=np.int32)

        # Sort instances by confidence (process high confidence first)
        sorted_indices = np.argsort(pred_scores)[::-1]

        for idx in sorted_indices:
            score = pred_scores[idx]
            if score < self.conf_threshold:
                continue

            class_id = pred_classes[idx]
            mask = mask_probs[idx] > 0.5  # Binary mask

            # Only update pixels where this mask is active
            seg_map[mask] = class_id

        # Resize segmentation map to original image size
        seg_map = cv2.resize(seg_map.astype(np.float32), (original_width, original_height),
                            interpolation=cv2.INTER_NEAREST).astype(np.uint8)

        return seg_map

    def _softmax(self, x, axis=-1):
        """Compute softmax values for array x along specified axis"""
        exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

    def create_colored_mask(self, seg_map):
        """
        Create a colored visualization of the segmentation map

        Args:
            seg_map (numpy.ndarray): Segmentation map with class IDs

        Returns:
            numpy.ndarray: Colored segmentation mask
        """
        height, width = seg_map.shape
        colored_mask = np.zeros((height, width, 3), dtype=np.uint8)

        # Assign colors to each class
        for class_id in range(min(len(self.class_names), seg_map.max() + 1)):
            mask = seg_map == class_id
            if np.any(mask):
                colored_mask[mask] = self.colors[class_id]

        return colored_mask

    def detect_and_draw(self, image, alpha=0.6, show_legend=True):
        """
        Run segmentation and draw results on image

        Args:
            image (numpy.ndarray): Input image in BGR format
            alpha (float): Transparency for overlay (0=transparent, 1=opaque)
            show_legend (bool): Whether to show class legend

        Returns:
            tuple: (segmentation_map, colored_mask, visualized_image)
        """
        # Preprocess
        input_tensor, orig_h, orig_w = self.preprocess(image)

        # Run inference
        outputs = self.session.run(self.output_names, {self.input_names[0]: input_tensor})

        # Postprocess
        seg_map = self.postprocess(outputs, orig_h, orig_w)

        # Create colored mask
        colored_mask = self.create_colored_mask(seg_map)

        # Blend with original image
        result = cv2.addWeighted(image, 1 - alpha, colored_mask, alpha, 0)

        # Get unique classes in the image
        unique_classes = np.unique(seg_map)
        unique_classes = unique_classes[unique_classes < len(self.class_names)]

        # Draw legend
        if show_legend and len(unique_classes) > 0:
            legend_height = min(30 * len(unique_classes), 400)
            legend_width = 200
            y_offset = 10

            for class_id in unique_classes:
                if class_id < len(self.class_names):
                    color = self.colors[class_id].tolist()
                    class_name = self.class_names[class_id]

                    # Draw colored box
                    cv2.rectangle(result, (10, y_offset), (40, y_offset + 20), color, -1)
                    cv2.rectangle(result, (10, y_offset), (40, y_offset + 20), (255, 255, 255), 1)

                    # Draw class name
                    cv2.putText(result, class_name, (45, y_offset + 15),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.putText(result, class_name, (45, y_offset + 15),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

                    y_offset += 25

        return seg_map, colored_mask, result

    def get_class_statistics(self, seg_map):
        """
        Get statistics about classes present in the segmentation

        Args:
            seg_map (numpy.ndarray): Segmentation map

        Returns:
            dict: Dictionary with class names and their pixel counts
        """
        unique_classes, counts = np.unique(seg_map, return_counts=True)

        stats = {}
        for class_id, count in zip(unique_classes, counts):
            if class_id < len(self.class_names):
                class_name = self.class_names[class_id]
                stats[class_name] = count

        # Sort by pixel count
        stats = dict(sorted(stats.items(), key=lambda x: x[1], reverse=True))

        return stats
