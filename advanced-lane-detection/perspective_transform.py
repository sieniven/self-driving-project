import os
import numpy as np
import cv2
import matplotlib.pyplot as plt


# MODIFY THIS FUNCTION TO GENERATE OUTPUT 
# THAT LOOKS LIKE THE IMAGE ABOVE
def corners_unwarp(undist):
    """
    function takes in a binary image and applies perspective transform to get a warped image.
    """
    # Convert binary image
    img_size = (undist.shape[1], undist.shape[0])

    # source points
    src = np.float32([[595, 450], [685,  450], [1000, 660], [280, 660]])
    # destination points
    dst = np.float32([[300, 0], [980, 0], [980, 720], [300, 720]])
    # calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    # warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(undist, M, img_size)

    # Return the resulting image and matrix
    return warped, Minv


# get top down perspective images using undistorted images
filenames = os.listdir(os.path.join(os.getcwd(), "advanced-lane-detection/output_images/undistorted/"))
for filename in filenames:
    img = cv2.imread(os.path.join(os.getcwd(), "advanced-lane-detection/output_images/undistorted/", filename))
    top_down, perspective_M = corners_unwarp(img)
    
    #save image
    cv2.imwrite(os.path.join(os.getcwd(), "advanced-lane-detection/output_images/perspective_transform", filename[:-4] + '_perspective.jpg'), top_down)

# get top down perspective images using binary images
filenames = os.listdir(os.path.join(os.getcwd(), "advanced-lane-detection/output_images/binary/"))
for filename in filenames:
    img = cv2.imread(os.path.join(os.getcwd(), "advanced-lane-detection/output_images/binary/", filename))
    top_down, perspective_M = corners_unwarp(img)
    
    #save image
    cv2.imwrite(os.path.join(os.getcwd(), "advanced-lane-detection/output_images/perspective_transform", filename[:-4] + '_perspective.jpg'), top_down)
