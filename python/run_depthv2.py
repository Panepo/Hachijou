# run_depthv2.py
import argparse
import time
import cv2
from datetime import datetime
from depthv2 import DepthAnythingV2
import numpy as np

# Global variables for mouse hover indicator
mouse_x, mouse_y = -1, -1
current_depth_map = None


def mouse_callback(event, x, y, flags, param):
    """Mouse callback to track cursor position"""
    global mouse_x, mouse_y
    if event == cv2.EVENT_MOUSEMOVE:
        mouse_x = x
        mouse_y = y


def draw_depth_indicator(image, depth_map, x, y, side_by_side=False):
    """
    Draw depth value indicator at mouse position

    Args:
        image: Display image to draw on
        depth_map: Depth map array
        x, y: Mouse coordinates
        side_by_side: Whether the display is side-by-side view
    """
    if x < 0 or y < 0:
        return image

    h, w = image.shape[:2]
    if y >= h or x >= w:
        return image

    # Adjust x coordinate if side-by-side view (depth is on the right half)
    depth_x = x
    if side_by_side:
        half_width = w // 2
        if x >= half_width:
            # Mouse is on depth side
            depth_x = x - half_width
        else:
            # Mouse is on original image side, map to depth coordinates
            depth_x = x

    # Scale mouse coordinates to depth map size
    depth_h, depth_w = depth_map.shape[:2]
    if side_by_side:
        scale_x = depth_w / (w // 2)
    else:
        scale_x = depth_w / w
    scale_y = depth_h / h

    depth_map_x = int(depth_x * scale_x)
    depth_map_y = int(y * scale_y)
def mouse_callback(event, x, y, flags, param):
    """Mouse callback to track cursor position"""
    global mouse_x, mouse_y
    if event == cv2.EVENT_MOUSEMOVE:
        mouse_x, mouse_y = x, y


def draw_depth_indicator(image, depth_map, x, y, side_by_side=False):
    """
    Draw depth value indicator at mouse position

    Args:
        image: Display image
        depth_map: Depth map array
        x, y: Mouse position
        side_by_side: Whether the image is side-by-side view

    Returns:
        Image with depth indicator drawn
    """
    if x < 0 or y < 0:
        return image

    # Adjust coordinates for side-by-side view
    depth_x = x
    if side_by_side:
        depth_x = x - image.shape[1] // 2
        if depth_x < 0:
            return image

    # Check bounds
    if 0 <= depth_x < depth_map.shape[1] and 0 <= y < depth_map.shape[0]:
        depth_value = depth_map[y, depth_x]

        # Draw crosshair
        cv2.drawMarker(image, (x, y), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)

        # Draw depth value text
        text = f"Depth: {depth_value:.2f}"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        text_x = x + 15
        text_y = y - 15

        # Keep text in bounds
        if text_x + text_size[0] > image.shape[1]:
            text_x = x - text_size[0] - 15
        if text_y < text_size[1]:
            text_y = y + text_size[1] + 15

        # Draw text background
        cv2.rectangle(image, (text_x - 5, text_y - text_size[1] - 5),
                     (text_x + text_size[0] + 5, text_y + 5), (0, 0, 0), -1)
        # Draw text
        cv2.putText(image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                   0.6, (0, 255, 0), 2, cv2.LINE_AA)

    return image


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
    if box_x + text_w + 10 > w:
        box_x = x - text_w - 25
    if box_y - text_h - 10 < 0:
        box_y = y + 30

    # Draw background box
    cv2.rectangle(image,
                  (box_x - 5, box_y - text_h - 5),
                  (box_x + text_w + 5, box_y + baseline + 5),
                  (0, 0, 0), -1)
    cv2.rectangle(image,
                  (box_x - 5, box_y - text_h - 5),
                  (box_x + text_w + 5, box_y + baseline + 5),
                  (0, 255, 0), 1)

    # Draw text
    cv2.putText(image, text, (box_x, box_y), font, font_scale, (0, 255, 0), thickness, cv2.LINE_AA)

    return image


def parse_args():
    parser = argparse.ArgumentParser(description="Run Depth Anything V2 ONNX on image or webcam.")

    # Model settings
    parser.add_argument("--model", type=str, default="./models/depth_anything_v2_vits.onnx",
                       help="Path to Depth Anything V2 ONNX model.")
    parser.add_argument("--height", type=int, default=518, help="Model input height.")
    parser.add_argument("--width", type=int, default=518, help="Model input width.")
    parser.add_argument("--colormap", type=str, default="plasma",
                       choices=["inferno", "plasma", "viridis", "jet", "turbo", "hot", "cool", "gray"],
                       help="Colormap for depth visualization.")

    # Single image mode
    parser.add_argument("--image", type=str, help="Path to input image.")
    parser.add_argument("--output", type=str, default="depth_output.jpg",
                       help="Path to save output image.")

    # Webcam / video mode
    parser.add_argument("--webcam", action="store_true", help="Use webcam as input.")
    parser.add_argument("--video", type=str, help="Path to input video file.")
    parser.add_argument("--cam-id", type=int, default=0, help="Webcam device ID.")
    parser.add_argument("--save-video", type=str, help="Path to save output video.")
    parser.add_argument("--screenshot-dir", type=str, default="./screenshots",
                       help="Directory to save screenshots (default: ./screenshots).")
    parser.add_argument("--no-show", action="store_true", help="Don't display video window.")
    parser.add_argument("--flip", action="store_true", help="Flip webcam horizontally.")
    parser.add_argument("--side-by-side", action="store_true",
                       help="Show original and depth side-by-side.")

    return parser.parse_args()


def get_colormap(name):
    """Convert colormap name to OpenCV colormap constant"""
    colormaps = {
        "inferno": cv2.COLORMAP_INFERNO,
        "plasma": cv2.COLORMAP_PLASMA,
        "viridis": cv2.COLORMAP_VIRIDIS,
        "jet": cv2.COLORMAP_JET,
        "turbo": cv2.COLORMAP_TURBO,
        "hot": cv2.COLORMAP_HOT,
        "cool": cv2.COLORMAP_COOL,
        "gray": cv2.COLORMAP_BONE,
    }
    return colormaps.get(name.lower(), cv2.COLORMAP_INFERNO)


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


def run_image(args):
    """Run Depth Anything V2 on a single image"""
    colormap = get_colormap(args.colormap)

    # Initialize depth estimator
    depth_estimator = DepthAnythingV2(
        model_path=args.model,
        input_size=(args.height, args.width)
    )
    depth_estimator.colormap = colormap

    # Load image
    image = cv2.imread(args.image)
    if image is None:
        raise RuntimeError(f"Failed to load image: {args.image}")

    print(f"Processing {args.image}...")
    t0 = time.time()

    # Predict depth
    depth_map, colored_depth = depth_estimator.predict_and_visualize(image, colormap)

    inference_time = (time.time() - t0) * 1000.0

    # Create output
    if args.side_by_side:
        output = depth_estimator.create_side_by_side(image, colored_depth)
    else:
        output = colored_depth

    # Save result
    import os
    os.makedirs("./output", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"./output/{timestamp}.jpg"
    cv2.imwrite(output_path, output)

    print(f"Saved depth map to {output_path}")
    print(f"Inference time: {inference_time:.1f}ms")
    print(f"Depth range: {depth_map.min():.2f} to {depth_map.max():.2f}")

    # Display result with mouse hover indicator
    global current_depth_map
    current_depth_map = depth_map

    window_name = "Depth Estimation"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    print("Hover mouse to see depth values. Press any key to close...")

    while True:
        display_img = output.copy()
        display_img = draw_depth_indicator(display_img, depth_map, mouse_x, mouse_y, args.side_by_side)
        cv2.imshow(window_name, display_img)

        key = cv2.waitKey(30) & 0xFF
        if key != 255:  # Any key pressed
            break

    cv2.destroyAllWindows()


def run_video(args):
    """Run Depth Anything V2 on video file"""
    colormap = get_colormap(args.colormap)

    # Initialize depth estimator
    depth_estimator = DepthAnythingV2(
        model_path=args.model,
        input_size=(args.height, args.width)
    )
    depth_estimator.colormap = colormap

    # Open video file
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video file: {args.video}")

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_read = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video opened: {args.video}")
    print(f"Resolution: {width}x{height} @ {fps_read:.1f} FPS")
    print(f"Total frames: {total_frames}")
    print(f"Model: {args.model}")
    print(f"Input size: {args.height}x{args.width}")
    print(f"Colormap: {args.colormap}")
    print("Press 's' to save screenshot, 'c' to change colormap, 'q' or ESC to exit, SPACE to pause.")

    # Video writer if saving
    writer = None
    if args.save_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_width = width * 2 if args.side_by_side else width
        writer = cv2.VideoWriter(args.save_video, fourcc, fps_read, (output_width, height))
        if not writer.isOpened():
            print(f"Warning: failed to open video writer at {args.save_video}")
            writer = None
        else:
            print(f"Saving video to {args.save_video}")

    # FPS tracking
    avg_infer_ms = 0.0
    alpha = 0.1  # smoothing factor for EMA
    frame_count = 0
    start_time = time.time()
    paused = False

    window_name = "Depth Anything V2 - Video"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret or frame is None:
                    print("End of video or frame grab failed.")
                    break

                # Inference
                t0 = time.time()
                depth_map, colored_depth = depth_estimator.predict_and_visualize(frame, colormap)
                infer_ms = (time.time() - t0) * 1000.0
                avg_infer_ms = alpha * infer_ms + (1 - alpha) * avg_infer_ms if avg_infer_ms > 0 else infer_ms

                # Calculate FPS
                frame_count += 1
                elapsed_time = time.time() - start_time
                fps_avg = frame_count / elapsed_time if elapsed_time > 0 else 0

                # Create visualization
                if args.side_by_side:
                    vis = depth_estimator.create_side_by_side(frame, colored_depth)
                else:
                    vis = colored_depth

                # Draw HUD (FPS and latency info)
                hud = f"FPS: {fps_avg:.1f}  Infer: {infer_ms:.1f}ms  Avg: {avg_infer_ms:.1f}ms"
                cv2.putText(vis, hud, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

                # Add frame progress info
                progress_info = f"Frame: {frame_count}/{total_frames}"
                cv2.putText(vis, progress_info, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

                # Add depth range info
                depth_info = f"Depth: {depth_map.min():.2f} - {depth_map.max():.2f}"
                cv2.putText(vis, depth_info, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

                # Draw depth indicator at mouse position
                vis = draw_depth_indicator(vis, depth_map, mouse_x, mouse_y, args.side_by_side)

                # Save video frame
                if writer is not None:
                    writer.write(vis)

            # Show window
            if not args.no_show:
                if paused:
                    pause_text = "PAUSED - Press SPACE to resume"
                    cv2.putText(vis, pause_text, (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

                cv2.imshow(window_name, vis)

                # Exit with q or ESC
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    print("Exit requested.")
                    break
                # Save screenshot with 's' key
                elif key == ord('s'):
                    screenshot_path = save_screenshot(vis, args.screenshot_dir)
                    print(f"\n📸 Screenshot saved: {screenshot_path}")
                # Change colormap with 'c' key
                elif key == ord('c'):
                    colormaps_list = [
                        cv2.COLORMAP_INFERNO, cv2.COLORMAP_PLASMA, cv2.COLORMAP_VIRIDIS,
                        cv2.COLORMAP_JET, cv2.COLORMAP_TURBO, cv2.COLORMAP_HOT
                    ]
                    current_idx = colormaps_list.index(colormap) if colormap in colormaps_list else 0
                    colormap = colormaps_list[(current_idx + 1) % len(colormaps_list)]
                    depth_estimator.colormap = colormap
                    print(f"Changed colormap to index {colormaps_list.index(colormap)}")
                # Pause/resume with SPACE
                elif key == ord(' '):
                    paused = not paused
                    print(f"{'Paused' if paused else 'Resumed'}")

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
        print(f"  Average inference time: {avg_infer_ms:.1f}ms")


def run_webcam(args):
    """Run Depth Anything V2 on webcam stream"""
    colormap = get_colormap(args.colormap)

    # Initialize depth estimator
    depth_estimator = DepthAnythingV2(
        model_path=args.model,
        input_size=(args.height, args.width)
    )
    depth_estimator.colormap = colormap

    # Open webcam
    cap = cv2.VideoCapture(args.cam_id)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open webcam with id {args.cam_id}")

    # Get camera properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_read = cap.get(cv2.CAP_PROP_FPS) or 30.0

    print(f"Webcam opened: {width}x{height} @ ~{fps_read:.1f} FPS")
    print(f"Model: {args.model}")
    print(f"Input size: {args.height}x{args.width}")
    print(f"Colormap: {args.colormap}")
    print("Press 's' to save screenshot, 'c' to change colormap, 'q' or ESC to exit.")

    # Video writer if saving
    writer = None
    if args.save_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_width = width * 2 if args.side_by_side else width
        writer = cv2.VideoWriter(args.save_video, fourcc, fps_read, (output_width, height))
        if not writer.isOpened():
            print(f"Warning: failed to open video writer at {args.save_video}")
            writer = None
        else:
            print(f"Saving video to {args.save_video}")

    # FPS tracking
    avg_infer_ms = 0.0
    alpha = 0.1  # smoothing factor for EMA
    frame_count = 0
    start_time = time.time()

    window_name = "Depth Anything V2 - Webcam"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("Frame grab failed; stopping.")
                break

            if args.flip:
                frame = cv2.flip(frame, 1)

            # Inference
            t0 = time.time()
            depth_map, colored_depth = depth_estimator.predict_and_visualize(frame, colormap)
            infer_ms = (time.time() - t0) * 1000.0
            avg_infer_ms = alpha * infer_ms + (1 - alpha) * avg_infer_ms if avg_infer_ms > 0 else infer_ms

            # Calculate FPS
            frame_count += 1
            elapsed_time = time.time() - start_time
            fps_avg = frame_count / elapsed_time if elapsed_time > 0 else 0

            # Create visualization
            if args.side_by_side:
                vis = depth_estimator.create_side_by_side(frame, colored_depth)
            else:
                vis = colored_depth

            # Draw HUD (FPS and latency info)
            hud = f"FPS: {fps_avg:.1f}  Infer: {infer_ms:.1f}ms  Avg: {avg_infer_ms:.1f}ms"
            cv2.putText(vis, hud, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

            # Add depth range info
            depth_info = f"Depth: {depth_map.min():.2f} - {depth_map.max():.2f}"
            cv2.putText(vis, depth_info, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

            # Draw depth indicator at mouse position
            vis = draw_depth_indicator(vis, depth_map, mouse_x, mouse_y, args.side_by_side)

            # Show window
            if not args.no_show:
                cv2.imshow(window_name, vis)
                # Exit with q or ESC
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    print("Exit requested.")
                    break
                # Save screenshot with 's' key
                elif key == ord('s'):
                    screenshot_path = save_screenshot(vis, args.screenshot_dir)
                    print(f"\n📸 Screenshot saved: {screenshot_path}")
                # Change colormap with 'c' key
                elif key == ord('c'):
                    colormaps_list = [
                        cv2.COLORMAP_INFERNO, cv2.COLORMAP_PLASMA, cv2.COLORMAP_VIRIDIS,
                        cv2.COLORMAP_JET, cv2.COLORMAP_TURBO, cv2.COLORMAP_HOT
                    ]
                    current_idx = colormaps_list.index(colormap) if colormap in colormaps_list else 0
                    colormap = colormaps_list[(current_idx + 1) % len(colormaps_list)]
                    depth_estimator.colormap = colormap
                    print(f"Changed colormap to index {colormaps_list.index(colormap)}")

            # Save video frame
            if writer is not None:
                writer.write(vis)

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
        print(f"  Average inference time: {avg_infer_ms:.1f}ms")


def main():
    args = parse_args()

    # Priority: if --image is provided, run single-image mode
    if args.image:
        run_image(args)
        return

    # If --video is provided, run video mode
    if args.video:
        run_video(args)
        return

    # Otherwise, webcam mode if requested
    if args.webcam:
        run_webcam(args)
        return

    # If neither provided, show usage hint
    print("Nothing to run. Provide --image for single image, --video for video file, or --webcam for live camera.")
    print("\nExamples:")
    print("  Image:   python run_depthv2.py --image test.jpg --output depth_result.jpg --side-by-side")
    print("  Video:   python run_depthv2.py --video input.mp4 --save-video output.mp4 --side-by-side")
    print("  Webcam:  python run_depthv2.py --webcam --side-by-side")
    print("  Save:    python run_depthv2.py --webcam --save-video depth_output.mp4 --colormap plasma")


if __name__ == "__main__":
    main()
