# Advanced Lane Detection Project

## Project Outline

The outline of this project is:
* Compute camera calibration matrix and distortion coefficients using chessboard images
* Implement distortion coefficients to apply distortion correction to frames
* Use color transformation, gradient filtering (Sobel operator), and other image processing techniques to obtain thresholded binary frames
* Apply perspective transformation to rectify binary image and get "birds-eye-view"
* Detection of lane pixels to fit and find lane boundary
* Determine curvature of the line and vehicle position with respect to center of the lane
* Apply perspective transform and warp the detected lane boundaries back onto the original image
* Output software capabilities to visually display of the lane boundaries, with numerical estimation of lane curvature and vehicle position information in every frame


[//]: # (Image References)

[image1]: ./examples/undistort_output.png "Chessboard"
[image2]: ./test_images/straight_lines2.jpg "Distorted Image"
[image3]: ./output_images/undistorted/straight_lines2_undistorted.jpg "Undistorted Image"
[image4]: ./output_images/binary/straight_lines2_binary.jpg "Binary"
[image5]: ./output_images/perspective_transform/straight_lines2_undistorted_perspective.jpg "Warp Undistorted"
[image6]: ./output_images/perspective_transform/straight_lines2_binary_perspective.jpg "Warp Binary"
[image7]: ./examples/histogram.png "Histogram"
[image8]: ./output_images/test_output/straight_lines2_undistorted_output.jpg "Results"


---

## Methodology

### 1. Camera Calibration

![image1][image1]

The first step in achieving the lane line detection capability in this project is to calibrate our cameras. This step is required in order to obtain undistorted images, as seen in the image above. There are two types of distortions that occur in camera images: radial and tangential distortions.
* Radial distortion: distortions occur due to light rays being bent either too much or too little at the edges of a curved lens of the camera.
* Tangentail distortion: distortions occur as camera lens is not aligned perfectly parrallel to the imaging plane, resulting in images to look 'tilted'.

Thus, we calibrate our cameras using a set of chessboard images, and using the chessboard corners detected to get the distortion coefficients. By using object points of the chessboard corners, we can assume that the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image. We can then obtain image points, which are the pixel position of each of the corners in the image plane. Finally, we can calculate the distortion coefficients from by comparing these values.

Subsequently, we tested our distortion coefficients obtained with test images. For example, our initial distorted image can be noted below. 

![image2][image2]

We use the output 'objpoints' and 'imgpoints' to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function. Refer to the exact code here: [**calibrate_camera.py**](./calibrate_camera.py). We apply this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![image3][image3]

### 2. Image Processing

Subsequently, we can use image processing techniques such as color transformation and gradient filtering to obtain binary form of the undistorted image. This allows us to filter out noise and identify the lane lines better. In our processing methodology, we use color thresholding with LAB channels. Specifically, we use 'L' (lightness) and 'A' (Red/Green) channels. In addition, we applied Sobel operator in the x-direction for the 'L' channel to apply gradient filtering on the image. Finally, we obtaining the binary threshold image by combining the 3 processed images information together.

```python
binary[(5 <= sobelx) & ((225 <= l_channel) | (155 <= b_channel))] = 1
```

Afterwhich, we apply masks (top, left and right) of the binary image to filter out the noises caused from our surroundings. Thus, obtain a clean binary threshold image of only the road lane lines, with minimal noise. Refer to the exact code here: [**color_gradient_threshold.py**](./color_gradient_threshold.py)

The binary threshold image processing step was tested onto our undistorted images. An example of the output image is below:

![image4][image4]


### 3. Perspective Transformation

For lane line detection capabilities, it is important that we are able to change our perspective to view the same scene from a different viewpoint/angle. Thus, our next step is to apply perspective transformation of our image to better estimate lane lines.

We apply a perspective transform to change our viewpoint to a "bird's eye view" of the scene. Our top-down view is achieved by defining 4 source points (rectangle) on the road in the image, and warp the points into 4 destination points into our warpped image. The values of these points are below:

```python
src = np.float32(
    [[595, 450],
    [685, 450],
    [1000, 660],
    [280, 660]])

dst = np.float32(
    [[300, 0],
    [980, 0],
    [980, 720],
    [300, 720]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 595, 450      | 300, 0        | 
| 685, 450      | 980, 0        |
| 1000, 660     | 980, 720      |
| 280, 660      | 300, 720      |

We toggled the values of these arguments to get the must suitable points for the perspective transform. It was verified by drawing the 'src' and 'dst' points onto a test image and its warped counterpart to ensure that the lines appear parallel in the warped image. In the image below, we tested the perspective transformation on undistorted images.

![image5][image5]

Subsequently, we applied the transformation onto our binary thresholded images. An example can be seen below:

![image6][image6]

### 4. Sliding Window Algorithm

After getting our bird's eye view binary thresholded image of the road, our next step would be to identify the lane lines. To do this, we incorporate a sliding window technique. This is done by plotting a histogram initially of the binary activations occur across the image (add the pixel values along each column up). We get the two peaks corresponding to the left and right lanes, and use these points as starting points to search for the lines. An example of the histogram result will look like this:

![image7][image7]

Subsequently, we incorporate a sliding window placed around the line centers to find and follow the lines up to the top of the frame. We incorporate this algorithm and tested it on our test images. You may refer to the code at [**lane_line_detection.py**](./lane_line_detection.py). We obtain the following results.

![image8][image8]


### 5. Radius of Curvature and Vehicle Positioning

After which, we use the equation for radius of curvature to caluclate the value. We also convert the radius of curvature value from pixels space to real-world space (metres). This is done by projecting the conversion from pixels space to meters using our warped image. For our calculations, we use:

```python
# define conversions in x and y from pixels space to meters
ym_per_pix = 30/720 # meters per pixel in y dimension
xm_per_pix = 3.7/700 # meters per pixel in x dimension
``` 

Lastly, we calculated the vehicle distance with respect to the centre line of the lane. Refer to the code at [**lane_line_detection.py**](./lane_line_detection.py). Refer to the code at [**lane_line_detection.py**](./lane_line_detection.py).

### 6. Final Results

Finally, we combined everything by apply perspective transform and warpping the detected lane boundaries back onto the original image. The output of our software capabilities is tested on the test images so that it can visually display the lane boundaries, and have the numerical estimation of lane curvature and vehicle position information. An example of the final output results is as follows:

![image9][image9]

---

## Video Results

We consolidated all of the above capabilities and tested our algorithm on the project video.

* [Test video results.](./output_videos/project_video_output.mp4)
* [Test video results from top-view.](./output_videos/project_video_topview.mp4)

---

## Discussion

### 1. Difficulties faced in design and implementation of project

Some difficulties face with this project were fine tuning the various parameters to get desireable results. For example, it was a difficult and tedious process of configuring parameters for thresholding of color transformations and gradients. 

In addition, testing to see which combinations of image processing techniques that would yield the best binary threshold image results were also very tedious and difficult process. For instance, using color transformation of different color channels would produce different results that may be more beneficial than the other in different settings. Thus, finding the right combination of image processing was one of the biggest difficulties I faced doing this project.

Lastly, I felt that designing the algorithm to draw out the polynomial lines was a difficult process, as it involved averaging out previous frames' polynomial coefficient values, which made the algorithm more complex.

#### 2. Shortfalls in the software capabilities

One of the possible shortfall in the software capability is that when the road bends start to get more extreme, the algorithm will no longer work or be accurate.

In addition, upsloping / downward sloping roads will make the algorithm less accurate. This is because perspective transformation will be affected as the plane of the road is no longer flat.

Lastly, road conditions such as lighting and weather, will severely affect the algorithm. This is because it will affect the visibility of the road lines, and cause the detection of the lines to be difficult.

### 3. What could you do to make it more robust?

Using a deep learning algorithm that incorporates semantic segmentation of the roads, by differentiating between road and non-road for example, is possibly a much more robust method. 