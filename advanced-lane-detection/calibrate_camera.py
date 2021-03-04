import os
import numpy as np
import cv2
import matplotlib.pyplot as plt


def get_calibration(cal_dir):
    """
    function gets objectpoints and imgpoints, and draws ChessboardCorners
    """
    # arrays to store object points and image points from all the images
    objpoints = []  # 3D points in real world space
    imgpoints = []  # 2D points in image plane

    # prepare object points
    objp = np.zeros((6*9, 3), np.float32)
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)   # x, y coordinates

    # get list of chessboard images
    filenames = os.listdir(cal_dir)

    for filename in filenames:
        img = cv2.imread(os.path.join(cal_dir, filename))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find and draw the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

        if ret:
            imgpoints.append(corners)
            objpoints.append(objp)
            cv2.drawChessboardCorners(img, (9, 6), corners, ret)

    dist_pickle = {"objpoints": objpoints, "imgpoints": imgpoints}
    
    return dist_pickle


def calibrate_camera(img, dist_pickle):
    """
    function calibrates the camera to get undistorted image
    """
    # get objpoints and imgpoints
    objpoints = dist_pickle["objpoints"]
    imgpoints = dist_pickle["imgpoints"]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    undist = cv2.undistort(img, mtx, dist, None, mtx)

    return undist


# run functions to calibrate camera, and get undistorted images for test images
cal_dir = os.path.join(os.getcwd(), "advanced-lane-detection/camera_cal")
dist_pickle = get_calibration(cal_dir)

test_dir = os.path.join(os.getcwd(), "advanced-lane-detection/test_images")
filenames = os.listdir(test_dir)
# read test chessboard images
for filename in filenames:
    img = cv2.imread(os.path.join(test_dir, filename))
    undist = calibrate_camera(img, dist_pickle)
    #save image
    cv2.imwrite(os.path.join(os.getcwd(), "advanced-lane-detection/output_images/undistorted", filename[:-4] + '_undistorted.jpg'), undist)