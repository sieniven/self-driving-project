# **Finding Lane Lines on the Road** 

## Project

### Background Information

This project is to develop the software capabilities to detect lane lines in images and videos. Lanes act as our constant reference for where to steer the vehicle. Naturally, it is one of the first and most important things we would like to develop in a self-driving car software algorithm.

In this project, I had to first develop a pipeline for the lane finding algorithm. Subsequently, I developed the software algorithm that is able to detect lane line with sample videos.

---

### Results

**1. Lane Detection Images**

The results of the sample images tested with the pipeline I developed is as follows.

* ![solid-white-curve sample image][image1]
* ![solid-white-right image][image2]
* ![solid-yellow-curve image][image3]
* ![solid-yellow-curve-2 image][image4]
* ![solid-yellow-left image][image5]
* ![white-car-lane-switch image][image6]


I tested my pipeline with the test images before testing them with the video samples. The images are as follows: 

[image1]: ./test_images/solidWhiteCurveprocessed.png (solid-white-curve sample image)
[image2]: ./test_images/solidWhiteRightprocessed.png (solid-white-right image)
[image3]: ./test_images/solidYellowCurveprocessed.png (solid-yellow-curve image)
[image4]: ./test_images/solidYellowCurve2processed.png (solid-yellow-curve-2 image)
[image5]: ./test_images/solidYellowLeftprocessed.png (solid-yellow-left imaged)
[image6]: ./test_images/whiteCarLaneSwitchprocessed.png (white-car-lane-switch image)

**2. Lane Detection Sample Videos**

The results of the sample videos tested with the lane-line detection algorithm I developed can be found in ./test_video_output/ folder.

---

### Reflection

**1. Algorithm/Pipeline design**

The design of the lane-line detection algorithm design consists of 5 steps. 
* Convert the image/frame to grayscale
* Apply Gaussian blurring
* Apply Canny Edge detector to obtain edges from the image
* Apply a polygon mask to only obtain the regions of interest in the frame/image that contains the road lines
* Apply Hough transformation to identify and connect the lines from the detected edges

**2. Drawing of lines to extrapolate line segments**

In order to draw a single line on the left and right lanes after applying Hough transform, the draw_lines() function is created such that it calculates the average line equation (mx + b) for both the left and right lanes. The line segments can be distinguished between the two sides if their gradients are positive or negative. Thus, the function calculates the average values of m and b the line segments on the left and right lines, and then draws the lines into the frame/image.

**3. Potential shortcomings with current algorithm***

A possible shortcoming can arise when objects other than the lines appear inside the defined polygon masked area. As the edge detection algorithm and subsequent Hough transformation is applied within the masked area, the presence of these objects will cause their edges to be detected as well. This will disrupt the line segments obtained, and even cause inaccurate markings into the frames.

Another shortcoming could be when there are sharp bends in the roads or slopes in these roads. The roads may end up out of the area of the defined polygon masked area, thus causing the lane lines detected to be of a much shorter range. 

### 3. Suggest possible improvements to your pipeline

A possible improvement to this algorithm would be to include a filtering algorithm to filter out noises in the edges detected.