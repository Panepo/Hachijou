# run_combo.py
import argparse
import time
import cv2
import numpy as np
from yolov3 import YOLOv3ONNX
from depthv2 import DepthAnythingV2


def parse_args():
    parser = argparse.ArgumentParser(description="Run YOLOv3 + Depth Anything V2 for person distance estimation.")

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

    # Input settings
    parser.add_argument("--image", type=str, help="Path to input image.")
    parser.add_argument("--output", type=str, default="combo_output.jpg", help="Path to save output image.")
    parser.add_argument("--webcam", action="store_true", help="Use webcam as input.")
    parser.add_argument("--cam-id", type=int, default=0, help="Webcam device ID.")
    parser.add_argument("--flip", action="store_true", help="Flip webcam horizontally.")

    # Visualization settings
    parser.add_argument("--show-depth", action="store_true", help="Show depth map alongside detection.")
    parser.add_argument("--save-video", type=str, help="Path to save output video.")
    parser.add_argument("--no-show", action="store_true", help="Don't display video window.")
    parser.add_argument("--colormap", type=str, default="plasma",
                       choices=["inferno", "plasma", "viridis", "jet", "turbo"],
                       help="Colormap for depth visualization.")

    return parser.parse_args()


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


def draw_person_with_distance(image, detection, depth_value, yolo):
    """
    Draw bounding box with distance information for a person

    Args:
        image: Image to draw on
        detection: [class_id, confidence, x1, y1, x2, y2]
        depth_value: Depth/distance value
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

    # Prepare label with distance
    label = f"{class_name}: {confidence:.2f}"
    distance_label = f"Depth: {depth_value:.2f}"

    # Draw label background
    (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    (dist_w, dist_h), dist_baseline = cv2.getTextSize(distance_label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)

    max_width = max(label_w, dist_w)
    total_height = label_h + dist_h + baseline + dist_baseline + 10

    cv2.rectangle(image, (int(x1), int(y1) - total_height - 5),
                 (int(x1) + max_width + 10, int(y1)), color, -1)

    # Draw labels
    cv2.putText(image, label, (int(x1) + 5, int(y1) - dist_h - dist_baseline - 5),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(image, distance_label, (int(x1) + 5, int(y1) - 5),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    return image


def process_frame(frame, yolo, depth_estimator, colormap, show_depth=False):
    """
    Process a single frame with both models

    Args:
        frame: Input frame
        yolo: YOLOv3ONNX instance
        depth_estimator: DepthAnythingV2 instance
        colormap: OpenCV colormap for depth visualization
        show_depth: Whether to show depth map

    Returns:
        tuple: (output_image, num_persons, avg_depth_time, avg_yolo_time)
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
    for detection in detections:
        class_id = int(detection[0])
        if class_id == 0:  # Person class
            # Get depth value for this person
            depth_value = get_person_depth(detection, depth_map)

            # Draw person with distance
            output = draw_person_with_distance(output, detection, depth_value, yolo)
            person_count += 1

    # Combine with depth map if requested
    if show_depth:
        # Resize depth map to match frame height
        h, w = frame.shape[:2]
        colored_depth_resized = cv2.resize(colored_depth, (w, h))
        output = np.hstack([output, colored_depth_resized])

    return output, person_count, depth_time, yolo_time


def run_image(args, yolo, depth_estimator, colormap):
    """Process single image"""
    # Load image
    image = cv2.imread(args.image)
    if image is None:
        raise RuntimeError(f"Failed to load image: {args.image}")

    print(f"Processing {args.image}...")

    # Process frame
    output, person_count, depth_time, yolo_time = process_frame(
        image, yolo, depth_estimator, colormap, args.show_depth
    )

    total_time = depth_time + yolo_time

    # Save result
    cv2.imwrite(args.output, output)

    print(f"Saved result to {args.output}")
    print(f"Found {person_count} person(s)")
    print(f"YOLOv3 time: {yolo_time:.1f}ms")
    print(f"Depth time: {depth_time:.1f}ms")
    print(f"Total time: {total_time:.1f}ms")

    # Display result
    cv2.imshow("Person Distance Detection", output)
    print("Press any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def run_webcam(args, yolo, depth_estimator, colormap):
    """Process webcam stream"""
    # Open webcam
    cap = cv2.VideoCapture(args.cam_id)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open webcam with id {args.cam_id}")

    # Get camera properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_read = cap.get(cv2.CAP_PROP_FPS) or 30.0

    print(f"Webcam opened: {width}x{height} @ ~{fps_read:.1f} FPS")
    print("Press 'q' or ESC to exit.")

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

    window_name = "Person Distance Detection"

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
            output, person_count, depth_time, yolo_time = process_frame(
                frame, yolo, depth_estimator, colormap, args.show_depth
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

            # Show window
            if not args.no_show:
                cv2.imshow(window_name, output)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
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
    print(f"YOLOv3 will detect persons (showing distance for each)")
    print(f"Depth map colormap: {args.colormap}\n")

    # Run appropriate mode
    if args.image:
        run_image(args, yolo, depth_estimator, colormap)
    elif args.webcam:
        run_webcam(args, yolo, depth_estimator, colormap)
    else:
        print("Nothing to run. Provide --image or --webcam.")
        print("\nExamples:")
        print("  Image:   python run_combo.py --image test.jpg --output result.jpg")
        print("  Webcam:  python run_combo.py --webcam --show-depth")
        print("  Save:    python run_combo.py --webcam --save-video output.mp4 --show-depth")


if __name__ == "__main__":
    main()
