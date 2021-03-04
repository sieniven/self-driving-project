import os
import numpy as np
import cv2
import matplotlib.pyplot as plt


def color_gradient_threshold(img):
    """
    function takes in undistorted images and create thresholded binary images using color transformation and gradients
    """
    # convert to HLS color space and separate the S channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    l_channel = hls[:,:,1]
    img_size = (l_channel.shape[1], l_channel.shape[0])

    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0, ksize=5) # Take the derivative in x
    sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    sobelx = np.uint8(255*sobelx/np.max(sobelx))

    # color thresholding
    l_channel = cv2.cvtColor(img, cv2.COLOR_BGR2LUV)[:,:,0]
    b_channel = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[:,:,2]

    # create a binary image
    binary = np.zeros_like(l_channel)
    binary[(5 <= sobelx) & ((225 <= l_channel) | (155 <= b_channel))] = 1

    # apply masks
    left_mask = [[0, 430], [0, 720], [150, 720], [560, 430]]
    right_mask = [[740, 430], [1150, 720], [1280, 720], [1280, 430]]
    center_mask = [[0, 0], [0, 430], [1280, 430], [1280, 0]]
    cv2.fillPoly(binary, np.int_([left_mask]), 0)
    cv2.fillPoly(binary, np.int_([right_mask]), 0)
    cv2.fillPoly(binary, np.int_([center_mask]), 0)

    return binary


filenames = os.listdir(os.path.join(os.getcwd(), "advanced-lane-detection/output_images/undistorted/"))
for filename in filenames:
    img = cv2.imread(os.path.join(os.getcwd(), "advanced-lane-detection/output_images/undistorted/", filename))
    binary = color_gradient_threshold(img)
    # save images
    cv2.imwrite(os.path.join(os.getcwd(), "advanced-lane-detection/output_images/binary", filename[:-16] + '_binary.jpg'), np.dstack((binary, binary, binary)).astype(np.uint8) * 255)