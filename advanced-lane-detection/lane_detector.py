import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from lane_detection import Lanes
from moviepy.editor import VideoFileClip


def process_image(lanes, img):
    # get undist image, binary threshold image, and perspective transform image
    lanes.get_binary_warped(img)
    # fit polynomial into the frame
    lanes.fit_polynomial()

    return lanes.result, lanes.out_img, lanes.binary_warped, lanes.color_warp


def test_images():
    filenames = os.listdir(os.path.join(os.getcwd(), "advanced-lane-detection/output_images/undistorted/"))
    for filename in filenames:
        img = cv2.imread(os.path.join(os.getcwd(), "advanced-lane-detection/output_images/undistorted/", filename))
        lanes = Lanes()
        lanes.get_calibration()
        result, out_img, binary_warped, color_warp = process_image(lanes, img)
        #save image
        cv2.imwrite(os.path.join(os.getcwd(), "advanced-lane-detection/output_images/test_output", filename[:-4] + '_output.jpg'), result)  
        cv2.imwrite(os.path.join(os.getcwd(), "advanced-lane-detection/output_images/test_output", filename[:-4] + '_color_warp.jpg'), color_warp)  
        cv2.imwrite(os.path.join(os.getcwd(), "advanced-lane-detection/output_images/test_output", filename[:-4] + '_out.jpg'), out_img)  
        plt.figure()
        plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.show()


def video(filename):
    # initialize lanes object and calibrate camera
    lanes = Lanes()
    lanes.get_calibration()
    
    # get videopath
    videoname = os.path.join(os.getcwd(), "advanced-lane-detection/", filename)
    output = os.path.join(os.getcwd(), "advanced-lane-detection/output_videos", "video_output.mp4")
    top_view_output = os.path.join(os.getcwd(), "advanced-lane-detection/output_videos", "video_output_topview.mp4")
   
   # start video capture
    cap = cv2.VideoCapture(videoname)
    recording = cv2.VideoWriter(output, cv2.VideoWriter_fourcc(*'MP4V'), 30.0, (1280, 720))
    topview = cv2.VideoWriter(top_view_output, cv2.VideoWriter_fourcc(*'MP4V'), 30.0, (1280, 720))
    if cap.isOpened():
        print(f"Video capture successful!")
    else:
        print(f"Video capture unsuccessful!")
        cap.release()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            result, out_img, binary_warped, color_warp = process_image(lanes, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        else:
            break

        recording.write(result)
        topview.write(color_warp)
        if color_warp is not None:
            cv2.imshow(f"Output", cv2.cvtColor(lanes.binary * 255, cv2.COLOR_BGR2RGB))
            # cv2.imshow(f"Output", result)
            # cv2.imshow(f"Output", out_img)
            # cv2.imshow(f"Output", color_warp)

        lanes.frame_count += 1
    
    cap.release()
    recording.release()
    topview.release()
    cv2.destroyAllWindows()
    

def test_image():
    img = cv2.imread(os.path.join(os.getcwd(), "advanced-lane-detection/output_images/undistorted/straight_lines1_undistorted.jpg"))
    lanes = Lanes()
    lanes.get_calibration()
    result, out_img, binary_warped, color_warp = process_image(lanes, img)
    plt.figure()
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.figure()
    plt.imshow(out_img)
    plt.figure()
    plt.imshow(binary_warped)
    plt.figure()
    plt.imshow(color_warp)
    plt.show()    

# test_image()
test_images()
# video("challenge_video.mp4")