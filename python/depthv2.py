import cv2
import numpy as np
import onnxruntime as ort


class DepthAnythingV2:
    """Depth Anything V2 depth estimation class using ONNX Runtime"""

    def __init__(self, model_path, input_size=(518, 518)):
        """
        Initialize Depth Anything V2 model

        Args:
            model_path (str): Path to ONNX model file
            input_size (tuple): Model input size (height, width)
        """
        self.model_path = model_path
        self.input_size = input_size

        # Initialize ONNX Runtime session
        self.session = ort.InferenceSession(
            model_path,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )

        # Get model input and output names
        self.input_names = [input.name for input in self.session.get_inputs()]
        self.output_names = [output.name for output in self.session.get_outputs()]

        print(f"Model inputs: {self.input_names}")
        print(f"Model outputs: {self.output_names}")

        # Color map for visualization
        self.colormap = cv2.COLORMAP_INFERNO

    def preprocess(self, image):
        """
        Preprocess image for Depth Anything V2 model

        Args:
            image (numpy.ndarray): Input image in BGR format

        Returns:
            tuple: Preprocessed image tensor and original size
        """
        original_height, original_width = image.shape[:2]

        # Resize to model input size
        resized = cv2.resize(image, (self.input_size[1], self.input_size[0]),
                            interpolation=cv2.INTER_LINEAR)

        # Convert BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # Normalize using ImageNet mean and std
        mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
        std = np.array([58.395, 57.12, 57.375], dtype=np.float32)

        normalized = (rgb.astype(np.float32) - mean) / std

        # Transpose to CHW format and add batch dimension
        tensor = np.transpose(normalized, (2, 0, 1))
        tensor = np.expand_dims(tensor, axis=0).astype(np.float32)

        return tensor, (original_height, original_width)

    def postprocess(self, depth_output, original_size):
        """
        Postprocess model output to get depth map

        Args:
            depth_output: Model output depth map
            original_size (tuple): Original image size (height, width)

        Returns:
            numpy.ndarray: Depth map resized to original image size
        """
        # Remove batch dimension if present
        if len(depth_output.shape) == 4:
            depth_map = depth_output[0, 0]
        elif len(depth_output.shape) == 3:
            depth_map = depth_output[0]
        else:
            depth_map = depth_output

        # Resize to original image size
        depth_map = cv2.resize(depth_map, (original_size[1], original_size[0]),
                              interpolation=cv2.INTER_LINEAR)

        return depth_map

    def predict(self, image):
        """
        Predict depth map for an image

        Args:
            image (numpy.ndarray): Input image in BGR format

        Returns:
            numpy.ndarray: Depth map (same size as input image)
        """
        # Preprocess image
        input_tensor, original_size = self.preprocess(image)

        # Run inference
        input_feed = {self.input_names[0]: input_tensor}
        outputs = self.session.run(self.output_names, input_feed)

        # Postprocess output
        depth_map = self.postprocess(outputs[0], original_size)

        return depth_map

    def visualize_depth(self, depth_map, colormap=None):
        """
        Visualize depth map with color mapping

        Args:
            depth_map (numpy.ndarray): Depth map
            colormap: OpenCV colormap (default: COLORMAP_INFERNO)

        Returns:
            numpy.ndarray: Colored depth visualization
        """
        if colormap is None:
            colormap = self.colormap

        # Normalize depth map to 0-255 range
        depth_min = depth_map.min()
        depth_max = depth_map.max()

        if depth_max - depth_min > 0:
            normalized = ((depth_map - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)
        else:
            normalized = np.zeros_like(depth_map, dtype=np.uint8)

        # Apply colormap
        colored = cv2.applyColorMap(normalized, colormap)

        return colored

    def predict_and_visualize(self, image, colormap=None):
        """
        Predict depth and return visualization

        Args:
            image (numpy.ndarray): Input image in BGR format
            colormap: OpenCV colormap for visualization

        Returns:
            tuple: (depth_map, colored_depth_visualization)
        """
        depth_map = self.predict(image)
        colored = self.visualize_depth(depth_map, colormap)

        return depth_map, colored

    def create_side_by_side(self, image, depth_colored):
        """
        Create side-by-side comparison of original image and depth map

        Args:
            image (numpy.ndarray): Original image
            depth_colored (numpy.ndarray): Colored depth visualization

        Returns:
            numpy.ndarray: Side-by-side comparison image
        """
        # Ensure both images have the same height
        h1, w1 = image.shape[:2]
        h2, w2 = depth_colored.shape[:2]

        if h1 != h2:
            depth_colored = cv2.resize(depth_colored, (w2, h1))

        # Concatenate horizontally
        combined = np.hstack([image, depth_colored])

        return combined


# Example usage
if __name__ == "__main__":
    # Initialize depth estimator
    depth_estimator = DepthAnythingV2(
        model_path="models/depth_anything_v2_vits.onnx",
        input_size=(518, 518)
    )

    # Load and process an image
    image = cv2.imread("test_image.jpg")
    if image is not None:
        # Predict depth
        depth_map, colored_depth = depth_estimator.predict_and_visualize(image)

        # Create side-by-side comparison
        result = depth_estimator.create_side_by_side(image, colored_depth)

        # Display results
        cv2.imshow("Depth Estimation", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Save depth map
        cv2.imwrite("depth_map.png", colored_depth)
        print(f"Depth map saved to depth_map.png")
        print(f"Depth range: {depth_map.min():.2f} to {depth_map.max():.2f}")
    else:
        print("Error: Could not load image")
