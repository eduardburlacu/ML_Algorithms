import numpy as np
import cv2
import sys

# Define the checkerboard dimensions
CHECKERBOARD = (6, 9)  # Adjust to your checkerboard dimensions
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points (0,0,0), (1,0,0), ..., (6,8,0)
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[1], 0:CHECKERBOARD[0]].T.reshape(-1, 2)

# Arrays to store object points and image points
objpoints = []  # 3D points in real world space
imgpoints = []  # 2D points in image plane

# Open the camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    sys.exit()

print("Press 'q' to quit and calibrate.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    if ret:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)
        frame = cv2.drawChessboardCorners(frame, CHECKERBOARD, corners2, ret)

    cv2.imshow('Camera', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Perform camera calibration
if len(objpoints) > 0 and len(imgpoints) > 0:
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    if ret:
        print("Camera matrix:\n", camera_matrix)
        print("Distortion coefficients:\n", dist_coeffs)

        # Compute projection matrix
        rmat, _ = cv2.Rodrigues(rvecs[0])
        proj_matrix = np.hstack((rmat, tvecs[0]))
        print("Projection matrix:\n", proj_matrix)

        # Decompose projection matrix
        _, cam_matrix, rot_matrix, trans_vec, _, _, _ = cv2.decomposeProjectionMatrix(proj_matrix)
        print("Decomposed camera matrix:\n", cam_matrix)
        print("Rotation matrix:\n", rot_matrix)
        print("Translation vector:\n", trans_vec)
    else:
        print("Calibration failed.")
else:
    print("Not enough data for calibration.")
