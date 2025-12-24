import cv2
import numpy as np
import json
import argparse

def calibrate_camera_and_save():
    """Calibrate camera using checkerboard pattern and save parameters"""
    # Checkerboard dimensions (internal corners)
    CHECKERBOARD = (9, 6)  # 9x6 internal corners (10x7 squares)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Prepare object points (0,0,0), (1,0,0), (2,0,0) ...
    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

    # Arrays to store object points and image points
    objpoints = []  # 3D points in real world space
    imgpoints = []  # 2D points in image plane

    # Capture checkerboard images from webcam
    cap = cv2.VideoCapture(0)
    image_count = 0
    required_images = 20  # Number of calibration images

    print("=" * 60)
    print("CAMERA CALIBRATION")
    print("=" * 60)
    print("Instructions:")
    print("1. Hold a printed checkerboard pattern (9x6 corners)")
    print("2. Move it to different positions and angles")
    print("3. Press SPACE when pattern is detected to capture")
    print(f"4. Capture {required_images} images")
    print("5. Press ESC to finish early")
    print("=" * 60)

    while image_count < required_images:
        ret, frame = cap.read()
        if not ret:
            print("Error: Cannot read from camera")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Find checkerboard corners
        ret_corners, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

        # Draw and display corners
        display_frame = frame.copy()
        if ret_corners:
            cv2.drawChessboardCorners(display_frame, CHECKERBOARD, corners, ret_corners)
            cv2.putText(display_frame, f"Pattern found! Press SPACE ({image_count}/{required_images})",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(display_frame, "Move checkerboard into view",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.putText(display_frame, f"Images captured: {image_count}/{required_images}",
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        cv2.imshow('Camera Calibration', display_frame)

        key = cv2.waitKey(1)
        if key == 27:  # ESC
            print("\nCalibration stopped by user")
            break
        elif key == 32 and ret_corners:  # SPACE
            # Refine corner positions
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            objpoints.append(objp)
            imgpoints.append(corners2)
            image_count += 1
            print(f"✓ Captured image {image_count}/{required_images}")

    cap.release()
    cv2.destroyAllWindows()

    if image_count < 10:
        print("\n❌ Not enough images for calibration!")
        print(f"   Only {image_count} images captured. Need at least 10.")
        return None

    # Perform calibration
    print("\n" + "=" * 60)
    print("Calibrating camera...")
    h, w = gray.shape[:2]
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, (w, h), None, None
    )

    # Extract parameters
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]

    # Calculate FOV
    fov_x = 2 * np.arctan2(w, 2 * fx) * 180 / np.pi
    fov_y = 2 * np.arctan2(h, 2 * fy) * 180 / np.pi
    fov_diagonal = 2 * np.arctan2(np.sqrt(w**2 + h**2), 2 * np.sqrt(fx**2 + fy**2)) * 180 / np.pi

    # Calculate reprojection error
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i],
                                         camera_matrix, dist_coeffs)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error
    mean_error /= len(objpoints)

    # Prepare parameters dictionary
    camera_params = {
        'calibration_date': '2025-12-23',
        'num_images_used': int(image_count),
        'image_width': int(w),
        'image_height': int(h),
        'camera_matrix': camera_matrix.tolist(),
        'distortion_coefficients': dist_coeffs.tolist(),
        'focal_length_x': float(fx),
        'focal_length_y': float(fy),
        'principal_point_x': float(cx),
        'principal_point_y': float(cy),
        'fov_horizontal_degrees': float(fov_x),
        'fov_vertical_degrees': float(fov_y),
        'fov_diagonal_degrees': float(fov_diagonal),
        'reprojection_error': float(mean_error)
    }

    # Save to JSON
    json_filename = 'camera_calibration.json'
    with open(json_filename, 'w') as f:
        json.dump(camera_params, f, indent=4)

    # Save to NumPy format (for easy loading in Python)
    npz_filename = 'camera_calibration.npz'
    np.savez(npz_filename,
             camera_matrix=camera_matrix,
             dist_coeffs=dist_coeffs,
             image_size=np.array([w, h]))

    # Print results
    print("=" * 60)
    print("✓ CALIBRATION COMPLETE")
    print("=" * 60)
    print(f"Images used: {image_count}")
    print(f"Image Size: {w}x{h} pixels")
    print(f"Focal Length (fx): {fx:.2f} pixels")
    print(f"Focal Length (fy): {fy:.2f} pixels")
    print(f"Principal Point: ({cx:.2f}, {cy:.2f})")
    print(f"Horizontal FOV: {fov_x:.2f}°")
    print(f"Vertical FOV: {fov_y:.2f}°")
    print(f"Diagonal FOV: {fov_diagonal:.2f}°")
    print(f"Reprojection Error: {mean_error:.4f} pixels")
    print("=" * 60)
    print(f"✓ Saved to: {json_filename}")
    print(f"✓ Saved to: {npz_filename}")
    print("=" * 60)

    return camera_params


def load_camera_params(json_file='camera_calibration.json'):
    """Load saved camera parameters from JSON file"""
    try:
        with open(json_file, 'r') as f:
            params = json.load(f)

        camera_matrix = np.array(params['camera_matrix'])
        dist_coeffs = np.array(params['distortion_coefficients'])

        return camera_matrix, dist_coeffs, params
    except FileNotFoundError:
        print(f"Error: {json_file} not found. Please run calibration first.")
        return None, None, None


def load_camera_params_npz(npz_file='camera_calibration.npz'):
    """Load camera parameters from NumPy format"""
    try:
        data = np.load(npz_file)
        return data['camera_matrix'], data['dist_coeffs'], data['image_size']
    except FileNotFoundError:
        print(f"Error: {npz_file} not found. Please run calibration first.")
        return None, None, None


def test_calibration():
    """Test calibration by undistorting live camera feed"""
    camera_matrix, dist_coeffs, params = load_camera_params()

    if camera_matrix is None:
        return

    print("\n" + "=" * 60)
    print("Testing calibration - showing undistorted camera feed")
    print(f"FOV: {params['fov_horizontal_degrees']:.1f}° x {params['fov_vertical_degrees']:.1f}°")
    print("Press ESC to exit")
    print("=" * 60)

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Undistort the image
        undistorted = cv2.undistort(frame, camera_matrix, dist_coeffs)

        # Display side by side
        combined = np.hstack([frame, undistorted])

        # Add labels
        cv2.putText(combined, "Original", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(combined, "Undistorted", (frame.shape[1] + 10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Calibration Test', combined)

        if cv2.waitKey(1) == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='Camera Calibration Tool')
    parser.add_argument('--test', action='store_true',
                       help='Test existing calibration (show undistorted view)')
    parser.add_argument('--load', action='store_true',
                       help='Load and display calibration parameters')

    args = parser.parse_args()

    if args.test:
        test_calibration()
    elif args.load:
        camera_matrix, dist_coeffs, params = load_camera_params()
        if params:
            print("\n" + "=" * 60)
            print("LOADED CALIBRATION PARAMETERS")
            print("=" * 60)
            print(f"Image Size: {params['image_width']}x{params['image_height']}")
            print(f"Horizontal FOV: {params['fov_horizontal_degrees']:.2f}°")
            print(f"Vertical FOV: {params['fov_vertical_degrees']:.2f}°")
            print(f"Focal Length: {params['focal_length_x']:.2f} pixels")
            print(f"Reprojection Error: {params['reprojection_error']:.4f} pixels")
            print("=" * 60)
    else:
        calibrate_camera_and_save()


if __name__ == "__main__":
    main()
