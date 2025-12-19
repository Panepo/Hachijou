import cv2
import numpy as np
import onnxruntime as ort


class MiDaSv2ONNX:
    """MiDaS v2 Depth Estimation Class using ONNX Runtime"""

    def __init__(self, model_path, input_size=(256, 256)):
        """
        Initialize MiDaS v2 depth estimator

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

        # Print model info for debugging
        print(f"Model loaded: {model_path}")
        print(f"Model inputs: {self.input_names}")
        print(f"Model outputs: {self.output_names}")
        print(f"Input size: {self.input_size}")

    def preprocess(self, image):
        """
        Preprocess image for MiDaS model

        Args:
            image (numpy.ndarray): Input image in BGR format

        Returns:
            numpy.ndarray: Preprocessed image tensor
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize to model input size
        resized = cv2.resize(image_rgb, (self.input_size[1], self.input_size[0]),
                           interpolation=cv2.INTER_CUBIC)

        # Normalize to [0, 1]
        normalized = resized.astype(np.float32) / 255.0

        # MiDaS normalization (ImageNet mean and std)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        normalized = (normalized - mean) / std

        # Transpose to CHW format and add batch dimension
        input_tensor = np.transpose(normalized, (2, 0, 1))
        input_tensor = np.expand_dims(input_tensor, axis=0)

        return input_tensor

    def postprocess(self, depth_map, original_shape):
        """
        Postprocess depth map to original image size

        Args:
            depth_map (numpy.ndarray): Raw depth map from model
            original_shape (tuple): Original image shape (height, width)

        Returns:
            numpy.ndarray: Processed depth map in original size
        """
        # Remove batch dimension
        if len(depth_map.shape) == 4:
            depth_map = depth_map.squeeze(0)
        if len(depth_map.shape) == 3:
            depth_map = depth_map.squeeze(0)

        # Resize to original image size
        depth_resized = cv2.resize(depth_map, (original_shape[1], original_shape[0]),
                                  interpolation=cv2.INTER_CUBIC)

        return depth_resized

    def normalize_depth(self, depth_map):
        """
        Normalize depth map to 0-255 range for visualization

        Args:
            depth_map (numpy.ndarray): Raw depth map

        Returns:
            numpy.ndarray: Normalized depth map (uint8)
        """
        # Normalize to [0, 1]
        depth_min = depth_map.min()
        depth_max = depth_map.max()

        if depth_max - depth_min > 1e-6:
            depth_normalized = (depth_map - depth_min) / (depth_max - depth_min)
        else:
            depth_normalized = np.zeros_like(depth_map)

        # Scale to [0, 255]
        depth_uint8 = (depth_normalized * 255).astype(np.uint8)

        return depth_uint8

    def estimate(self, image):
        """
        Estimate depth map from input image

        Args:
            image (numpy.ndarray): Input image in BGR format

        Returns:
            numpy.ndarray: Depth map (float32, original image size)
        """
        # Store original shape
        original_shape = image.shape[:2]

        # Preprocess image
        input_tensor = self.preprocess(image)

        # Run inference
        input_feed = {self.input_names[0]: input_tensor}
        outputs = self.session.run(self.output_names, input_feed)

        # Get depth map (first output)
        depth_map = outputs[0]

        # Postprocess to original size
        depth_map = self.postprocess(depth_map, original_shape)

        return depth_map

    def estimate_and_visualize(self, image, colormap=cv2.COLORMAP_INFERNO):
        """
        Estimate depth and create visualization

        Args:
            image (numpy.ndarray): Input image in BGR format
            colormap (int): OpenCV colormap for visualization

        Returns:
            tuple: (depth_map, depth_colored, combined_view)
                - depth_map: Raw depth map (float32)
                - depth_colored: Colored depth visualization (BGR)
                - combined_view: Side-by-side original and depth
        """
        # Estimate depth
        depth_map = self.estimate(image)

        # Normalize for visualization
        depth_uint8 = self.normalize_depth(depth_map)

        # Apply colormap
        depth_colored = cv2.applyColorMap(depth_uint8, colormap)

        # Create side-by-side view
        combined = np.hstack([image, depth_colored])

        return depth_map, depth_colored, combined

    def create_point_cloud(self, image, depth_map, fx=None, fy=None, cx=None, cy=None):
        """
        Create 3D point cloud from depth map (optional advanced feature)

        Args:
            image (numpy.ndarray): Original image
            depth_map (numpy.ndarray): Depth map
            fx, fy: Focal lengths (if None, estimated from image size)
            cx, cy: Principal point (if None, use image center)

        Returns:
            numpy.ndarray: Point cloud [N, 6] (x, y, z, r, g, b)
        """
        height, width = depth_map.shape

        # Default camera intrinsics if not provided
        if fx is None:
            fx = width * 0.7  # Approximate focal length
        if fy is None:
            fy = height * 0.7
        if cx is None:
            cx = width / 2
        if cy is None:
            cy = height / 2

        # Create coordinate grids
        u, v = np.meshgrid(np.arange(width), np.arange(height))

        # Calculate 3D coordinates
        z = depth_map
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy

        # Get RGB colors
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Flatten and combine
        points = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=1)
        colors = image_rgb.reshape(-1, 3)

        # Combine points and colors
        point_cloud = np.hstack([points, colors])

        return point_cloud


# Example usage
if __name__ == "__main__":
    # Initialize depth estimator
    estimator = MiDaSv2ONNX(
        model_path="models/midasv2.onnx",
        input_size=(384, 384)
    )

    # Load and estimate depth on an image
    image = cv2.imread("test_image.jpg")
    if image is not None:
        depth_map, depth_colored, combined = estimator.estimate_and_visualize(image)

        # Print depth statistics
        print(f"Depth map shape: {depth_map.shape}")
        print(f"Depth range: [{depth_map.min():.2f}, {depth_map.max():.2f}]")
        print(f"Mean depth: {depth_map.mean():.2f}")

        # Display results
        cv2.imshow("MiDaS v2 Depth Estimation", combined)
        cv2.imshow("Depth Map", depth_colored)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Optionally save results
        cv2.imwrite("depth_colored.jpg", depth_colored)
        cv2.imwrite("depth_combined.jpg", combined)
        print("Results saved!")
    else:
        print("Error: Could not load image")
