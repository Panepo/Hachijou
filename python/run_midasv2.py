
# run_midasv2.py
import argparse
import time
import cv2
import numpy as np
from datetime import datetime
from midasv2 import MiDaSv2ONNX

def parse_args():
    parser = argparse.ArgumentParser(description="Run MiDaS v2 ONNX depth estimation on image or webcam/video.")
    # Model settings
    parser.add_argument("--model", type=str, default="./models/midasv2.onnx",
                       help="Path to MiDaS v2 ONNX model.")
    parser.add_argument("--height", type=int, default=256,
                       help="Model input height (e.g., 256).")
    parser.add_argument("--width", type=int, default=256,
                       help="Model input width (e.g., 256).")
    parser.add_argument("--colormap", type=str, default="inferno",
                       choices=['inferno', 'viridis', 'plasma', 'magma', 'turbo', 'jet', 'hot', 'cool'],
                       help="Colormap for depth visualization.")

    # Single image mode
    parser.add_argument("--image", type=str,
                       help="Path to input image (if provided, runs single-image inference).")
    parser.add_argument("--output", type=str, default="depth_output.jpg",
                       help="Path to save output image.")
    parser.add_argument("--save-depth", type=str,
                       help="Path to save raw depth map as numpy file (.npy).")

    # Webcam / video mode
    parser.add_argument("--webcam", action="store_true",
                       help="Use webcam as input.")
    parser.add_argument("--cam-id", type=int, default=0,
                       help="Webcam device ID (0 is default camera).")
    parser.add_argument("--save-video", type=str,
                       help="Path to save output video.")
    parser.add_argument("--screenshot-dir", type=str, default="./screenshots",
                       help="Directory to save screenshots (default: ./screenshots).")
    parser.add_argument("--no-show", action="store_true",
                       help="Don't display video window.")
    parser.add_argument("--flip", action="store_true",
                       help="Flip webcam horizontally.")
    parser.add_argument("--view-mode", type=str, default="combined",
                       choices=['combined', 'depth', 'original'],
                       help="Display mode: combined (side-by-side), depth (only depth), original (with overlay)")

    return parser.parse_args()

def get_colormap(name):
    """Get OpenCV colormap from name"""
    colormaps = {
        'inferno': cv2.COLORMAP_INFERNO,
        'viridis': cv2.COLORMAP_VIRIDIS,
        'plasma': cv2.COLORMAP_PLASMA,
        'magma': cv2.COLORMAP_MAGMA,
        'turbo': cv2.COLORMAP_TURBO,
        'jet': cv2.COLORMAP_JET,
        'hot': cv2.COLORMAP_HOT,
        'cool': cv2.COLORMAP_COOL
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
    """Run MiDaS depth estimation on a single image"""
    # Initialize estimator
    midas = MiDaSv2ONNX(
        model_path=args.model,
        input_size=(args.height, args.width)
    )

    # Load image
    image = cv2.imread(args.image)
    if image is None:
        raise RuntimeError(f"Failed to load image: {args.image}")

    # Estimate depth
    print(f"Running depth estimation on {args.image}...")
    t0 = time.time()
    colormap = get_colormap(args.colormap)
    depth_map, depth_colored, combined = midas.estimate_and_visualize(image, colormap=colormap)
    infer_time = (time.time() - t0) * 1000.0

    # Save results
    import os
    os.makedirs("./output", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"./output/{timestamp}.jpg"
    cv2.imwrite(output_path, combined)
    print(f"Saved combined view to {output_path}")

    # Save depth colored separately
    depth_only_path = f"./output/{timestamp}_depth.jpg"
    cv2.imwrite(depth_only_path, depth_colored)
    print(f"Saved depth visualization to {depth_only_path}")

    # Save raw depth map if requested
    if args.save_depth:
        np.save(args.save_depth, depth_map)
        print(f"Saved raw depth map to {args.save_depth}")

    print(f"Inference time: {infer_time:.1f}ms")
    print(f"Depth statistics:")
    print(f"  Min depth: {depth_map.min():.2f}")
    print(f"  Max depth: {depth_map.max():.2f}")
    print(f"  Mean depth: {depth_map.mean():.2f}")
    print(f"  Std depth: {depth_map.std():.2f}")

def run_webcam(args):
    """Run MiDaS depth estimation on webcam stream"""
    # Initialize estimator
    midas = MiDaSv2ONNX(
        model_path=args.model,
        input_size=(args.height, args.width)
    )

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
    print(f"View mode: {args.view_mode}")
    print("Press 's' to save screenshot, 'c' to cycle colormaps, 'v' to cycle view modes, 'q' or ESC to exit.")

    # Video writer if saving
    writer = None
    if args.save_video:
        # Determine output size based on view mode
        if args.view_mode == 'combined':
            out_width = width * 2
            out_height = height
        else:
            out_width = width
            out_height = height

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(args.save_video, fourcc, fps_read, (out_width, out_height))
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

    # Colormap cycling
    colormaps = ['inferno', 'viridis', 'plasma', 'magma', 'turbo', 'jet', 'hot', 'cool']
    colormap_idx = colormaps.index(args.colormap)

    # View mode cycling
    view_modes = ['combined', 'depth', 'original']
    view_mode_idx = view_modes.index(args.view_mode)

    # Mouse hover tracking
    mouse_x, mouse_y = -1, -1

    def mouse_callback(event, x, y, flags, param):
        nonlocal mouse_x, mouse_y
        if event == cv2.EVENT_MOUSEMOVE:
            mouse_x, mouse_y = x, y

    # Set mouse callback
    if not args.no_show:
        cv2.namedWindow("MiDaS v2 Depth Estimation")
        cv2.setMouseCallback("MiDaS v2 Depth Estimation", mouse_callback)

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
            colormap = get_colormap(colormaps[colormap_idx])
            depth_map, depth_colored, combined = midas.estimate_and_visualize(frame, colormap=colormap)
            infer_ms = (time.time() - t0) * 1000.0
            avg_infer_ms = alpha * infer_ms + (1 - alpha) * avg_infer_ms if avg_infer_ms > 0 else infer_ms

            # Calculate FPS
            frame_count += 1
            elapsed_time = time.time() - start_time
            fps_avg = frame_count / elapsed_time if elapsed_time > 0 else 0

            # Select view based on mode
            current_view_mode = view_modes[view_mode_idx]
            if current_view_mode == 'combined':
                vis = combined
            elif current_view_mode == 'depth':
                vis = depth_colored
            else:  # 'original'
                # Overlay depth on original with transparency
                vis = cv2.addWeighted(frame, 0.6, depth_colored, 0.4, 0)

            # Draw HUD (FPS and latency info)
            hud = f"FPS: {fps_avg:.1f}  Infer: {infer_ms:.1f}ms  Avg: {avg_infer_ms:.1f}ms"
            info = f"Colormap: {colormaps[colormap_idx]}  View: {current_view_mode}"
            cv2.putText(vis, hud, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(vis, info, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

            # Depth statistics
            depth_stats = f"Depth: [{depth_map.min():.1f}, {depth_map.max():.1f}] mean={depth_map.mean():.1f}"
            cv2.putText(vis, depth_stats, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

            # Draw mouse hover depth indicator
            if mouse_x >= 0 and mouse_y >= 0:
                # Determine actual depth map coordinates based on view mode
                current_view_mode = view_modes[view_mode_idx]
                depth_x, depth_y = mouse_x, mouse_y

                if current_view_mode == 'combined':
                    # In combined view, depth is on the right half
                    if mouse_x >= width:
                        depth_x = mouse_x - width
                    else:
                        # Mouse is on original image side
                        depth_x = mouse_x
                        depth_y = mouse_y

                # Ensure coordinates are within bounds
                if 0 <= depth_x < depth_map.shape[1] and 0 <= depth_y < depth_map.shape[0]:
                    depth_value = depth_map[depth_y, depth_x]

                    # Draw crosshair at mouse position
                    cv2.drawMarker(vis, (mouse_x, mouse_y), (0, 255, 0),
                                 cv2.MARKER_CROSS, 20, 2)

                    # Draw depth value tooltip
                    tooltip = f"Depth: {depth_value:.2f}"
                    (tw, th), _ = cv2.getTextSize(tooltip, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)

                    # Position tooltip near mouse, adjust if near edges
                    tooltip_x = mouse_x + 15
                    tooltip_y = mouse_y - 10

                    if tooltip_x + tw > vis.shape[1]:
                        tooltip_x = mouse_x - tw - 15
                    if tooltip_y - th < 0:
                        tooltip_y = mouse_y + 30

                    # Draw tooltip background
                    cv2.rectangle(vis,
                                (tooltip_x - 5, tooltip_y - th - 5),
                                (tooltip_x + tw + 5, tooltip_y + 5),
                                (0, 0, 0), -1)
                    cv2.rectangle(vis,
                                (tooltip_x - 5, tooltip_y - th - 5),
                                (tooltip_x + tw + 5, tooltip_y + 5),
                                (0, 255, 0), 2)

                    # Draw tooltip text
                    cv2.putText(vis, tooltip, (tooltip_x, tooltip_y),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

            # Show window
            if not args.no_show:
                cv2.imshow("MiDaS v2 Depth Estimation", vis)
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # q or ESC
                    print("Exit requested.")
                    break
                elif key == ord('s'):  # Save screenshot
                    screenshot_path = save_screenshot(vis, args.screenshot_dir)
                    print(f"\n📸 Screenshot saved: {screenshot_path}")
                elif key == ord('c'):  # Cycle colormaps
                    colormap_idx = (colormap_idx + 1) % len(colormaps)
                    print(f"Colormap changed to: {colormaps[colormap_idx]}")
                elif key == ord('v'):  # Cycle view modes
                    view_mode_idx = (view_mode_idx + 1) % len(view_modes)
                    print(f"View mode changed to: {view_modes[view_mode_idx]}")

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

    # Priority: if --image is provided, run single-image mode.
    if args.image:
        run_image(args)
        return

    # Otherwise, webcam mode if requested.
    if args.webcam:
        run_webcam(args)
        return

    # If neither provided, show usage hint.
    print("Nothing to run. Provide --image for single image, or --webcam for live camera.")
    print("\nExamples:")
    print("  Image:   python run_midasv2.py --image test.jpg --output depth_result.jpg")
    print("  Webcam:  python run_midasv2.py --webcam --colormap viridis")
    print("  Save:    python run_midasv2.py --webcam --save-video depth_output.mp4")
    print("  Custom:  python run_midasv2.py --webcam --view-mode depth --colormap turbo")

if __name__ == "__main__":
    main()
