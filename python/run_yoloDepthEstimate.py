# run_yoloDepthEstimate.py
import argparse
import time
import math
import cv2
import numpy as np
from datetime import datetime
from yolov3 import YOLOv3ONNX
from depthv2 import DepthAnythingV2


def parse_args():
    parser = argparse.ArgumentParser(description="Run YOLOv3 + Depth Anything V2 for person distance estimation with real distance calculation.")

    # Model settings
    parser.add_argument("--yolo-model", type=str, default="./models/yolov3-12.onnx",
                       help="Path to YOLOv3 ONNX model.")
    parser.add_argument("--depth-model", type=str, default="./models/depth_anything_v2_vits.onnx",
                       help="Path to Depth Anything V2 ONNX model.")
    parser.add_argument("--yolo-size", type=int, nargs=2, default=[416, 416],
                       help="YOLOv3 input size [height width].")
    parser.add_argument("--depth-size", type=int, nargs=2, default=[518, 518],
                       help="Depth model input size [height width].")
    parser.add_argument("--conf", type=float, default=0.5, help="YOLOv3 confidence threshold.")
    parser.add_argument("--nms", type=float, default=0.4, help="YOLOv3 NMS threshold.")

    # Distance estimation settings
    parser.add_argument("--person-height", type=float, default=180.0,
                       help="Assumed person height in cm (default: 180.0cm).")
    parser.add_argument("--focal-length", type=float, default=None,
                       help="Camera focal length in pixels (auto-calibrate if not provided).")

    # Input settings
    parser.add_argument("--image", type=str, help="Path to input image.")
    parser.add_argument("--output", type=str, default="distance_output.jpg", help="Path to save output image.")
    parser.add_argument("--webcam", action="store_true", help="Use webcam as input.")
    parser.add_argument("--cam-id", type=int, default=0, help="Webcam device ID.")
    parser.add_argument("--flip", action="store_true", help="Flip webcam horizontally.")

    # Visualization settings
    parser.add_argument("--show-depth", action="store_true", help="Show depth map alongside detection.")
    parser.add_argument("--save-video", type=str, help="Path to save output video.")
    parser.add_argument("--no-show", action="store_true", help="Don't display video window.")
    parser.add_argument("--screenshot-dir", type=str, default="./screenshots",
                       help="Directory to save screenshots (default: ./screenshots).")
    parser.add_argument("--colormap", type=str, default="plasma",
                       choices=["inferno", "plasma", "viridis", "jet", "turbo"],
                       help="Colormap for depth visualization.")
    parser.add_argument("--calibrate", action="store_true",
                       help="Calibrate focal length (stand at known distance and press 'c').")
    parser.add_argument("--calibrate-distance", type=float, default=200.0,
                       help="Known distance for calibration in cm (default: 200.0cm).")

    return parser.parse_args()


def estimate_focal_length(image_width, image_height, fov_horizontal=60.0):
    """
    Estimate focal length based on typical webcam field of view

    Most webcams have a horizontal FOV between 60-75 degrees.
    This provides a reasonable estimate for distance calculation.

    Formula: focal_length = (image_width / 2) / tan(FOV / 2)

    Args:
        image_width: Image width in pixels
        image_height: Image height in pixels
        fov_horizontal: Horizontal field of view in degrees (default: 60)

    Returns:
        float: Estimated focal length in pixels
    """
    # Calculate focal length from horizontal FOV
    focal_length_x = (image_width / 2) / math.tan(math.radians(fov_horizontal / 2))

    # Optionally calculate from vertical FOV (assuming square pixels)
    # Most cameras have aspect ratio 4:3 or 16:9
    aspect_ratio = image_width / image_height
    fov_vertical = 2 * math.atan(math.tan(math.radians(fov_horizontal / 2)) / aspect_ratio)
    focal_length_y = (image_height / 2) / math.tan(fov_vertical / 2)

    # Use average (they should be close for square pixels)
    focal_length = (focal_length_x + focal_length_y) / 2

    return focal_length


def save_screenshot(image, screenshot_dir="./screenshots"):
    """
    Save a screenshot with timestamp

    Args:
        image: Image to save
        screenshot_dir: Directory to save screenshots

    Returns:
        str: Path to saved screenshot
    """
    import os

    # Create directory if it doesn't exist
    os.makedirs(screenshot_dir, exist_ok=True)

    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"screenshot_{timestamp}.jpg"
    filepath = os.path.join(screenshot_dir, filename)

    # Save image
    cv2.imwrite(filepath, image)

    return filepath


def get_colormap(name):
    """Convert colormap name to OpenCV colormap constant"""
    colormaps = {
        "inferno": cv2.COLORMAP_INFERNO,
        "plasma": cv2.COLORMAP_PLASMA,
        "viridis": cv2.COLORMAP_VIRIDIS,
        "jet": cv2.COLORMAP_JET,
        "turbo": cv2.COLORMAP_TURBO,
    }
    return colormaps.get(name.lower(), cv2.COLORMAP_PLASMA)


def get_person_depth(detection, depth_map):
    """
    Get average depth value within a person's bounding box

    Args:
        detection: [class_id, confidence, x1, y1, x2, y2]
        depth_map: Depth map array

    Returns:
        float: Average depth value in the bounding box
    """
    _, _, x1, y1, x2, y2 = detection

    # Convert to integers and ensure valid bounds
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    h, w = depth_map.shape[:2]

    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w))
    y2 = max(0, min(y2, h))

    if x2 <= x1 or y2 <= y1:
        return 0.0

    # Extract bounding box region from depth map
    bbox_depth = depth_map[y1:y2, x1:x2]

    # Calculate average depth (you can also use median or center point)
    avg_depth = np.mean(bbox_depth)

    return avg_depth


def calculate_distance(bbox_height_px, person_height_cm, focal_length):
    """
    Calculate actual distance using similar triangles principle

    Distance = (Real Height × Focal Length) / Pixel Height

    Args:
        bbox_height_px: Height of bounding box in pixels
        person_height_cm: Actual person height in cm
        focal_length: Camera focal length in pixels

    Returns:
        float: Estimated distance in cm
    """
    if bbox_height_px <= 0 or focal_length is None or focal_length <= 0:
        return None

    distance = (person_height_cm * focal_length) / bbox_height_px
    return distance


def calibrate_focal_length(bbox_height_px, person_height_cm, known_distance_cm):
    """
    Calibrate focal length based on known distance

    Focal Length = (Pixel Height × Known Distance) / Real Height

    Args:
        bbox_height_px: Height of bounding box in pixels at known distance
        person_height_cm: Actual person height in cm
        known_distance_cm: Known distance to person in cm

    Returns:
        float: Calibrated focal length in pixels
    """
    if bbox_height_px <= 0:
        return None

    focal_length = (bbox_height_px * known_distance_cm) / person_height_cm
    return focal_length


def draw_person_with_distance(image, detection, depth_value, distance_cm, yolo):
    """
    Draw bounding box with distance information for a person

    Args:
        image: Image to draw on
        detection: [class_id, confidence, x1, y1, x2, y2]
        depth_value: Depth/distance value from depth map
        distance_cm: Calculated real distance in cm
        yolo: YOLOv3ONNX instance for class names and colors

    Returns:
        numpy.ndarray: Image with drawn detection
    """
    class_id, confidence, x1, y1, x2, y2 = detection
    class_id = int(class_id)

    # Get class name and color
    class_name = yolo.class_names[class_id] if class_id < len(yolo.class_names) else f"Class {class_id}"
    color = tuple(int(c) for c in yolo.colors[class_id])

    # Draw bounding box
    cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

    # Prepare labels
    label = f"{class_name}: {confidence:.2f}"
    depth_label = f"Depth: {depth_value:.2f}"

    # Add real distance if calculated
    if distance_cm is not None:
        if distance_cm >= 100:
            distance_label = f"Distance: {distance_cm / 100:.2f}m"
        else:
            distance_label = f"Distance: {distance_cm:.1f}cm"
    else:
        distance_label = "Distance: N/A"

    # Get text sizes for background
    labels = [label, depth_label, distance_label]
    text_sizes = [cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1) for lbl in labels]

    max_width = max([size[0][0] for size in text_sizes])
    total_height = sum([size[0][1] + size[1] for size in text_sizes]) + 15

    # Draw label background
    cv2.rectangle(image, (int(x1), int(y1) - total_height - 5),
                 (int(x1) + max_width + 10, int(y1)), color, -1)

    # Draw labels
    y_offset = int(y1) - 5
    for i in range(len(labels) - 1, -1, -1):
        cv2.putText(image, labels[i], (int(x1) + 5, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        y_offset -= text_sizes[i][0][1] + text_sizes[i][1] + 5

    return image


def process_frame(frame, yolo, depth_estimator, colormap, person_height_cm, focal_length, show_depth=False):
    """
    Process a single frame with both models

    Args:
        frame: Input frame
        yolo: YOLOv3ONNX instance
        depth_estimator: DepthAnythingV2 instance
        colormap: OpenCV colormap for depth visualization
        person_height_cm: Assumed person height in cm
        focal_length: Camera focal length for distance calculation
        show_depth: Whether to show depth map

    Returns:
        tuple: (output_image, num_persons, avg_depth_time, avg_yolo_time, person_detections)
    """
    # Run YOLOv3 detection
    t0 = time.time()
    detections = yolo.detect(frame)
    yolo_time = (time.time() - t0) * 1000.0

    # Run depth estimation
    t0 = time.time()
    depth_map, colored_depth = depth_estimator.predict_and_visualize(frame, colormap)
    depth_time = (time.time() - t0) * 1000.0

    # Create output image
    output = frame.copy()

    # Filter for persons only (class_id = 0) and draw with distance
    person_count = 0
    person_detections = []

    for detection in detections:
        class_id = int(detection[0])
        if class_id == 0:  # Person class
            # Get depth value for this person
            depth_value = get_person_depth(detection, depth_map)

            # Calculate real distance based on bounding box height
            _, _, x1, y1, x2, y2 = detection
            bbox_height = abs(y2 - y1)
            distance_cm = calculate_distance(bbox_height, person_height_cm, focal_length)

            # Draw person with distance
            output = draw_person_with_distance(output, detection, depth_value, distance_cm, yolo)
            person_count += 1

            # Store detection info
            person_detections.append({
                'bbox': [x1, y1, x2, y2],
                'bbox_height': bbox_height,
                'depth': depth_value,
                'distance_cm': distance_cm,
                'confidence': detection[1]
            })

    # Combine with depth map if requested
    if show_depth:
        # Resize depth map to match frame height
        h, w = frame.shape[:2]
        colored_depth_resized = cv2.resize(colored_depth, (w, h))
        output = np.hstack([output, colored_depth_resized])

    return output, person_count, depth_time, yolo_time, person_detections


def run_image(args, yolo, depth_estimator, colormap, focal_length):
    """Process single image"""
    # Load image
    image = cv2.imread(args.image)
    if image is None:
        raise RuntimeError(f"Failed to load image: {args.image}")

    # Auto-estimate focal length if not provided
    if focal_length is None:
        h, w = image.shape[:2]
        focal_length = estimate_focal_length(w, h)
        print(f"Auto-estimated focal length: {focal_length:.1f}px (based on {w}x{h} and 60° FOV)")

    print(f"Processing {args.image}...")

    # Process frame
    output, person_count, depth_time, yolo_time, person_detections = process_frame(
        image, yolo, depth_estimator, colormap, args.person_height, focal_length, args.show_depth
    )

    total_time = depth_time + yolo_time

    # Print detection details
    print(f"\nDetected {person_count} person(s):")
    for i, person in enumerate(person_detections, 1):
        if person['distance_cm'] is not None:
            dist_str = f"{person['distance_cm'] / 100:.2f}m" if person['distance_cm'] >= 100 else f"{person['distance_cm']:.1f}cm"
            print(f"  Person {i}: Distance = {dist_str}, Depth = {person['depth']:.2f}, Confidence = {person['confidence']:.2f}")
        else:
            print(f"  Person {i}: Distance = N/A, Depth = {person['depth']:.2f}, Confidence = {person['confidence']:.2f}")

    # Save result
    import os
    os.makedirs("./output", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"./output/{timestamp}.jpg"
    cv2.imwrite(output_path, output)

    print(f"\nSaved result to {output_path}")
    print(f"YOLOv3 time: {yolo_time:.1f}ms")
    print(f"Depth time: {depth_time:.1f}ms")
    print(f"Total time: {total_time:.1f}ms")

    # Display result
    cv2.imshow("Person Distance Estimation", output)
    print("\nPress any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def run_webcam(args, yolo, depth_estimator, colormap, focal_length):
    """Process webcam stream"""
    # Open webcam
    cap = cv2.VideoCapture(args.cam_id)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open webcam with id {args.cam_id}")

    # Get camera properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_read = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # Auto-estimate focal length if not provided
    if focal_length is None:
        focal_length = estimate_focal_length(width, height)
        print(f"Auto-estimated focal length: {focal_length:.1f}px (based on {width}x{height} and 60° FOV)")
        print("Tip: Use --calibrate to improve accuracy, or --focal-length to set manually")

    print(f"Webcam opened: {width}x{height} @ ~{fps_read:.1f} FPS")
    print(f"Person height assumption: {args.person_height}cm")
    print(f"Focal length: {focal_length:.1f}px")
    if args.calibrate:
        print(f"\nCalibration mode: Press 'c' when standing at {args.calibrate_distance}cm")
    print("Press 's' to save screenshot, 'q' or ESC to exit.")

    # Video writer if saving
    writer = None
    if args.save_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_width = width * 2 if args.show_depth else width
        writer = cv2.VideoWriter(args.save_video, fourcc, fps_read, (output_width, height))
        if not writer.isOpened():
            print(f"Warning: failed to open video writer at {args.save_video}")
            writer = None
        else:
            print(f"Saving video to {args.save_video}")

    # FPS tracking
    avg_total_ms = 0.0
    avg_yolo_ms = 0.0
    avg_depth_ms = 0.0
    alpha = 0.1  # smoothing factor
    frame_count = 0
    start_time = time.time()

    window_name = "Person Distance Estimation"

    # Mutable focal length for calibration
    current_focal_length = focal_length

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("Frame grab failed; stopping.")
                break

            if args.flip:
                frame = cv2.flip(frame, 1)

            # Process frame
            t0 = time.time()
            output, person_count, depth_time, yolo_time, person_detections = process_frame(
                frame, yolo, depth_estimator, colormap, args.person_height, current_focal_length, args.show_depth
            )
            total_time = (time.time() - t0) * 1000.0

            # Update averages
            avg_total_ms = alpha * total_time + (1 - alpha) * avg_total_ms if avg_total_ms > 0 else total_time
            avg_yolo_ms = alpha * yolo_time + (1 - alpha) * avg_yolo_ms if avg_yolo_ms > 0 else yolo_time
            avg_depth_ms = alpha * depth_time + (1 - alpha) * avg_depth_ms if avg_depth_ms > 0 else depth_time

            # Calculate FPS
            frame_count += 1
            elapsed_time = time.time() - start_time
            fps_avg = frame_count / elapsed_time if elapsed_time > 0 else 0

            # Draw HUD
            hud_y = 30
            cv2.putText(output, f"FPS: {fps_avg:.1f}  Total: {total_time:.0f}ms  Persons: {person_count}",
                       (10, hud_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

            hud_y += 30
            cv2.putText(output, f"YOLO: {yolo_time:.0f}ms  Depth: {depth_time:.0f}ms",
                       (10, hud_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

            # Show focal length status
            hud_y += 30
            if current_focal_length is not None:
                cv2.putText(output, f"Focal Length: {current_focal_length:.1f}px",
                           (10, hud_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
            else:
                cv2.putText(output, "Focal Length: Not calibrated (press 'c' to calibrate)",
                           (10, hud_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 128, 255), 2, cv2.LINE_AA)

            # Show window
            if not args.no_show:
                cv2.imshow(window_name, output)
                key = cv2.waitKey(1) & 0xFF

                # Handle screenshot
                if key == ord('s'):
                    screenshot_path = save_screenshot(output, args.screenshot_dir)
                    print(f"\n📸 Screenshot saved: {screenshot_path}")

                # Handle calibration
                elif key == ord('c') and args.calibrate and person_count > 0:
                    # Use the first detected person for calibration
                    person = person_detections[0]
                    calibrated_fl = calibrate_focal_length(
                        person['bbox_height'],
                        args.person_height,
                        args.calibrate_distance
                    )
                    if calibrated_fl is not None:
                        current_focal_length = calibrated_fl
                        print(f"\n✓ Focal length calibrated: {current_focal_length:.1f}px")
                        print(f"  (Based on person at {args.calibrate_distance}cm with height {person['bbox_height']:.1f}px)")
                    else:
                        print("\n✗ Calibration failed: invalid bounding box height")

                elif key == ord('q') or key == 27:
                    print("Exit requested.")
                    break

            # Save video frame
            if writer is not None:
                writer.write(output)

    finally:
        # Cleanup
        cap.release()
        if writer is not None:
            writer.release()
            print(f"Video saved to {args.save_video}")
        if not args.no_show:
            cv2.destroyAllWindows()

        # Print statistics
        total_time = time.time() - start_time
        print(f"\nSession statistics:")
        print(f"  Total frames: {frame_count}")
        print(f"  Total time: {total_time:.1f}s")
        print(f"  Average FPS: {frame_count / total_time:.1f}")
        print(f"  Average total time: {avg_total_ms:.1f}ms")
        print(f"  Average YOLOv3 time: {avg_yolo_ms:.1f}ms")
        print(f"  Average Depth time: {avg_depth_ms:.1f}ms")
        if current_focal_length is not None:
            print(f"  Final focal length: {current_focal_length:.1f}px")


def main():
    args = parse_args()

    print("Initializing models...")

    # Initialize YOLOv3
    yolo = YOLOv3ONNX(
        model_path=args.yolo_model,
        conf_threshold=args.conf,
        nms_threshold=args.nms,
        input_size=tuple(args.yolo_size)
    )

    # Initialize Depth Anything V2
    colormap = get_colormap(args.colormap)
    depth_estimator = DepthAnythingV2(
        model_path=args.depth_model,
        input_size=tuple(args.depth_size)
    )
    depth_estimator.colormap = colormap

    print("\nModels loaded successfully!")
    print(f"YOLOv3 will detect persons and estimate distance")
    print(f"Assumed person height: {args.person_height}cm")
    print(f"Depth map colormap: {args.colormap}")

    # Set focal length
    focal_length = args.focal_length
    if focal_length is not None:
        print(f"Using manual focal length: {focal_length}px")
    else:
        print("Focal length will be auto-estimated from camera resolution")
        if args.calibrate:
            print("Calibration mode enabled for improved accuracy")

    print()

    # Run appropriate mode
    if args.image:
        run_image(args, yolo, depth_estimator, colormap, focal_length)
    elif args.webcam:
        run_webcam(args, yolo, depth_estimator, colormap, focal_length)
    else:
        print("Nothing to run. Provide --image or --webcam.")
        print("\nExamples:")
        print("  Image:      python run_yoloDepthEstimate.py --image test.jpg --focal-length 800")
        print("  Webcam:     python run_yoloDepthEstimate.py --webcam --calibrate --show-depth")
        print("  Save:       python run_yoloDepthEstimate.py --webcam --save-video output.mp4 --focal-length 800")
        print("  Calibrate:  python run_yoloDepthEstimate.py --webcam --calibrate --calibrate-distance 200")
        print("\nNote: For best results, calibrate the focal length by standing at a known distance and pressing 'c'")


if __name__ == "__main__":
    main()
