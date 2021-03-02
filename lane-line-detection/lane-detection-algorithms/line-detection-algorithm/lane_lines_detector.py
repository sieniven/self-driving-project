import os
import math
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip
from IPython.display import HTML
%matplotlib inline

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    `vertices` should be a numpy array of integer points.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    function to average/extrapolate the line segments detected, and map out the full extent of the lane  
    """
    
    # initialize lists to hold line formula values
    bLeftValues     = []  # b of left lines
    bRightValues    = []  # b of Right lines
    mPositiveValues = []  # m of Left lines
    mNegitiveValues = []  # m of Right lines
    
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)
            
            # calculate the slope and intercept
            m = (y2-y1)/(x2-x1)
            b = y1 - x1*m
            
            # threshold to check for outliers
            if m >= 0 and (m < 0.2 or m > 0.8):
                continue
            elif m < 0 and (m < -0.8 or m > -0.2):
                continue
            
            # seperate positive line and negative line slopes
            if m > 0:
                mPositiveValues.append(m)
                bLeftValues.append(b)
            else:
                mNegitiveValues.append(m)
                bRightValues.append(b)
                
    # Get image shape and define y region of interest value
    imshape = img.shape
    y_max   = imshape[0] # lines initial point at bottom of image    
    y_min   = 310        # lines end point at top of ROI

    # Get the mean of all the lines values
    AvgPositiveM = mean(mPositiveValues)
    AvgNegitiveM = mean(mNegitiveValues)
    AvgLeftB     = mean(bLeftValues)
    AvgRightB    = mean(bRightValues)

    # use average slopes to generate line using ROI endpoints
    if AvgPositiveM != 0:
        x1_Left = (y_max - AvgLeftB)/AvgPositiveM
        y1_Left = y_max
        x2_Left = (y_min - AvgLeftB)/AvgPositiveM
        y2_Left = y_min
        
        if AvgNegitiveM != 0:
            x1_Right = (y_max - AvgRightB)/AvgNegitiveM
            y1_Right = y_max
            x2_Right = (y_min - AvgRightB)/AvgNegitiveM
            y2_Right = y_min

            # define average left and right lines
            cv2.line(img, (int(x1_Left), int(y1_Left)), (int(x2_Left), int(y2_Left)), color, thickness) #avg Left Line
            cv2.line(img, (int(x1_Right), int(y1_Right)), (int(x2_Right), int(y2_Right)), color, thickness) #avg Right Line

        
def mean(list):
    """
    calculate mean of list
    """
    return float(sum(list)) / max(len(list), 1)


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img


def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, γ)


def test_images:
        """
    the code will build the pipeline that will draw lane lines on test_images,
    then save them to the test_images_output directory
    """
    files = os.listdir("test_images/")
    for filename in files:
        # reading in an image
        image = mpimg.imread("test_images/" + filename)

        # convert image to grayscale
        gray = grayscale(image)

        # apply Gaussian blurring
        kernel_size = 3
        blur_gray = gaussian_blur(gray, kernel_size)

        # apply Canny transform
        low_threshold = 100
        high_threshold = 200
        edges = canny(blur_gray, low_threshold, high_threshold)

        # create masked edges image
        imshape = image.shape
        vertices = np.array([[(0,imshape[0]),(460, 310), (490, 310), (imshape[1],imshape[0])]], dtype=np.int32)
        masked_edges = region_of_interest(edges, vertices)

        # define the Hough transform parameters
        rho = 1 # distance resolution in pixels of the Hough grid
        theta = np.pi/180 # angular resolution in radians of the Hough grid
        threshold = 22     # minimum number of votes (intersections in Hough grid cell)
        min_line_length = 18 #minimum number of pixels making up a line
        max_line_gap = 1    # maximum gap in pixels between connectable line segments

        line_img = hough_lines(masked_edges, rho, theta, threshold, min_line_length, max_line_gap)

        # create color binary image to combine with line image
        color_edges = np.dstack((edges, edges, edges))

        # draw the lines on the initial image
        lines_edges = weighted_img(line_img, image)

        # show images
        plt.figure()
        plt.imshow(image, cmap='Greys_r')
        plt.figure()
        plt.imshow(masked_edges, cmap='Greys_r')
        plt.figure()
        plt.imshow(line_img, cmap='Greys_r')
        plt.figure()
        plt.imshow(lines_edges, cmap='Greys_r')
        color_edges = np.dstack((edges, edges, edges))

        # draw the lines on the initial image
        lines_edges = weighted_img(line_img, image)

        # show images
        plt.figure()
        plt.imshow(image, cmap='Greys_r')
        plt.figure()
        plt.imshow(masked_edges, cmap='Greys_r')
        plt.figure()
        plt.imshow(line_img, cmap='Greys_r')
        plt.figure()
        plt.imshow(lines_edges, cmap='Greys_r')


def process_image(image):
    """
    function draws lane lines on frame/image
    """
    # convert image to grayscale
    gray = grayscale(image)

    # apply Gaussian blurring
    kernel_size = 3
    blur_gray = gaussian_blur(gray, kernel_size)

    # apply Canny transform
    low_threshold = 100
    high_threshold = 200
    edges = canny(blur_gray, low_threshold, high_threshold)

    # create masked edges image
    imshape = image.shape
    vertices = np.array([[(0,imshape[0]),(460, 310), (490, 310), (imshape[1],imshape[0])]], dtype=np.int32)
    masked_edges = region_of_interest(edges, vertices)

    # define the Hough transform parameters
    rho = 1 # distance resolution in pixels of the Hough grid
    theta = np.pi/180 # angular resolution in radians of the Hough grid
    threshold = 22     # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 18 #minimum number of pixels making up a line
    max_line_gap = 1    # maximum gap in pixels between connectable line segments

    line_img = hough_lines(masked_edges, rho, theta, threshold, min_line_length, max_line_gap)

    # create color binary image to combine with line image
    color_edges = np.dstack((edges, edges, edges))

    # draw the lines on the initial image
    result = weighted_img(line_img, image)
    
    return result


def get_video(filename):
    white_output = 'test_videos_output/' + filename
    ## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
    ## To do so add .subclip(start_second,end_second) to the end of the line below
    ## Where start_second and end_second are integer values representing the start and end of the subclip
    ## You may also uncomment the following line for a subclip of the first 5 seconds
    ##clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)
    clip1 = VideoFileClip("test_videos/" + filename)
    white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
    %time white_clip.write_videofile(white_output, audio=False)

    HTML("""
        <video width="960" height="540" controls>
        <source src="{0}">
        </video>
        """.format(white_output))


if __name__ == "__main__":
    filename = input("Please key in filename: ")
    get_video(filename)