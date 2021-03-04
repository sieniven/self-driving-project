import os
import numpy as np
import cv2
from calibrate_camera import calibrate_camera, get_calibration
from color_gradient_threshold import color_gradient_threshold
from perspective_transform import corners_unwarp


# define a class to store the characteristics of the lane detection
class Lanes:
    def __init__(self):
        self.frame = None
        self.frame_count = 0
        self.undist = None
        self.binary = None
        self.binary_warped = None
        self.Minv = None
        self.out_img = None
        self.color_warp = None
        self.color_warp2 = None
        self.result = None
        self.FLAG = True

        # was both lines detected in the last iteration
        self.detected = False
        self.invisible_count = 0

        # the left and right lane curve points
        self.left_x = None
        self.left_y = None
        self.right_x = None
        self.right_y = None

        # the coefficients of the fitted polynomials
        self.left_fit = None
        self.right_fit = None

        # radii of curvature 
        self.left_rad_curve_metre = None
        self.left_rad_curve_pixel = None
        self.right_rad_curve_metre = None
        self.right_rad_curve_pixel = None

        # distance in meters of vehicle center from the line
        self.line_base_pos = None

        # store all left and right coefficients of the fitted polynomial
        self.all_left = [[],[],[]]
        self.all_right = [[],[],[]]

        # confidence levels
        self.left_conf = None
        self.right_conf = None

        # get objectpoints and imgpoints for camera calibration
        self.dist_pickle = None
        
        
    def get_calibration(self):
        self.dist_pickle = get_calibration(os.path.join(os.getcwd(), "advanced-lane-detection/camera_cal"))


    def get_binary_warped(self, img):
        self.frame = img
        # get undistorted img
        self.undist = calibrate_camera(self.frame, self.dist_pickle)
        # get binary threshold img
        self.binary = color_gradient_threshold(self.undist)
        # get perspective transformation img
        self.binary_warped, self.Minv = corners_unwarp(self.binary)


    def fit_polynomial(self):
        """
        main pipeline to fit polynomial into frames
        """
        # Find our lane pixels first
        if self.detected:
            self.invisible_count = 0
            self.search_around_poly()

        else:
            if self.frame_count != 0:
                self.invisible_count += 1

            if self.invisible_count > 5:
                self.all_left = [[],[],[]]
                self.all_right = [[],[],[]]
                self.invisible_count = 0
            
            self.find_lane_pixels()
        
        try:
            # Fit a second order polynomial to each using 'np.polyfit'
            self.left_fit = np.polyfit(self.left_y, self.left_x, 2)
            self.right_fit = np.polyfit(self.right_y, self.right_x, 2)
        except:
            self.FLAG = False
            pass

        if self.FLAG:
            # store fits
            self.all_left[0].append(self.left_fit[0])
            self.all_left[1].append(self.left_fit[1])
            self.all_left[2].append(self.left_fit[2])
            self.all_right[0].append(self.right_fit[0])
            self.all_right[1].append(self.right_fit[1])
            self.all_right[2].append(self.right_fit[2])

            # average the final past 10 accepted fits to produce best fit
            if len(self.all_right) > 10 and len(self.all_left) > 10:
                self.left_fit[0] = np.mean(self.all_left[0][-10:])
                self.left_fit[1] = np.mean(self.all_left[1][-10:])
                self.left_fit[2] = np.mean(self.all_left[2][-10:])
                self.right_fit[0] = np.mean(self.all_right[0][-10:])
                self.right_fit[1] = np.mean(self.all_right[1][-10:])
                self.right_fit[2] = np.mean(self.all_right[2][-10:])

            # Generate x and y values for plotting
            ploty = np.linspace(0, self.binary_warped.shape[0]-1, self.binary_warped.shape[0])
            left_fitx = self.left_fit[0]*ploty**2 + self.left_fit[1]*ploty + self.left_fit[2]
            right_fitx = self.right_fit[0]*ploty**2 + self.right_fit[1]*ploty + self.right_fit[2]

            # Create an image to draw the lines on
            warp_zero = np.zeros_like(self.binary_warped).astype(np.uint8)
            self.color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
            self.color_warp2 = np.dstack((warp_zero, warp_zero, warp_zero))

            # Recast the x and y points into usable format for cv2.fillPoly()
            pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
            pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
            pts = np.hstack((pts_left, pts_right))

            # Generate a polygon to illustrate the search window area
            # And recast the x and y points into usable format for cv2.fillPoly()
            margin = 10
            margin_c = 3
        
            left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
            left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
            left_line_pts = np.hstack((left_line_window1, left_line_window2))
            
            right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
            right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
            right_line_pts = np.hstack((right_line_window1, right_line_window2))

            center_line_window1 = np.array([np.transpose(np.vstack([(left_fitx + right_fitx)/2-margin_c, ploty]))])
            center_line_window2 = np.array([np.flipud(np.transpose(np.vstack([(left_fitx + right_fitx)/2 + margin_c, ploty])))])
            center_line_pts = np.hstack((center_line_window1, center_line_window2))

            self.out_img[self.left_y, self.left_x] = [255, 0, 0]
            self.out_img[self.right_y, self.right_x] = [0, 0, 255]
            
            # Draw the lane onto the warped blank image
            cv2.fillPoly(self.color_warp, np.int_([pts]), (0, 255, 0))
            cv2.fillPoly(self.color_warp2, np.int_([left_line_pts]), (255, 255, 255))
            cv2.fillPoly(self.color_warp2, np.int_([right_line_pts]), (255, 255, 255))
            cv2.fillPoly(self.color_warp2, np.int_([center_line_pts]), (255, 255, 0))
            self.color_warp = cv2.addWeighted(self.color_warp, 1.0, self.color_warp2, 1.0, 0)
            
            # Warp the blank back to original image space using inverse perspective matrix (Minv)
            img = self.undist
            img_size = (img.shape[1], img.shape[0])
            new_out = cv2.warpPerspective(self.color_warp, self.Minv, img_size) 
            # Combine the result with the original image
            self.result = cv2.addWeighted(self.undist, 1, new_out, 1.0, 0.0)
            # get radius of curvature
            self.measure_curvature(ploty)
        else:
            self.result = self.undist
            self.FLAG = True


    def find_lane_pixels(self):
        """
        function to find line using sliding windows
        """
        # Take a histogram of the bottom half of the image
        histogram = np.sum(self.binary_warped[self.binary_warped.shape[0]//2:,:], axis=0)
        # Create an output image to draw on and visualize the result
        self.out_img = np.dstack((self.binary_warped, self.binary_warped, self.binary_warped))
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]//2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # HYPERPARAMETERS
        # Choose the number of sliding windows
        nwindows = 9
        # Set the width of the windows +/- margin
        margin = 120
        # Set minimum number of pixels found to recenter window
        minpix = 60

        # Set height of windows - based on nwindows above and image shape
        window_height = np.int(self.binary_warped.shape[0]//nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = self.binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated later for each window in nwindows
        leftx_current = leftx_base
        rightx_current = rightx_base

        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = self.binary_warped.shape[0] - (window+1)*window_height
            win_y_high = self.binary_warped.shape[0] - window*window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            
            # Draw the windows on the visualization image
            cv2.rectangle(self.out_img,(win_xleft_low,win_y_low), (win_xleft_high,win_y_high),(0,255,0), 2) 
            cv2.rectangle(self.out_img,(win_xright_low,win_y_low), (win_xright_high,win_y_high),(0,255,0), 2) 
            
            # Identify the nonzero pixels in x and y within the window #
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
            
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:        
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices (previously was a list of lists of pixels)

        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        self.left_x = nonzerox[left_lane_inds]
        self.left_y = nonzeroy[left_lane_inds] 
        self.right_x = nonzerox[right_lane_inds]
        self.right_y = nonzeroy[right_lane_inds]
        self.detected = True

        # evaluate polynomial
        self.evaluate_polynomial(left_lane_inds, right_lane_inds)


    def search_around_poly(self):
        # HYPERPARAMETER
        # Choose the width of the margin around the previous polynomial to search
        # The quiz grader expects 100 here, but feel free to tune on your own!
        margin = 100

        # Grab activated pixels
        nonzero = self.binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        
        ### TO-DO: Set the area of search based on activated x-values ###
        ### within the +/- margin of our polynomial function ###
        ### Hint: consider the window areas for the similarly named variables ###
        ### in the previous quiz, but change the windows to our new search area ###
        left_lane_inds = ((nonzerox > (self.left_fit[0]*(nonzeroy**2) + self.left_fit[1]*nonzeroy + 
                        self.left_fit[2] - margin)) & (nonzerox < (self.left_fit[0]*(nonzeroy**2) + 
                        self.left_fit[1]*nonzeroy + self.left_fit[2] + margin)))
        right_lane_inds = ((nonzerox > (self.right_fit[0]*(nonzeroy**2) + self.right_fit[1]*nonzeroy + 
                        self.right_fit[2] - margin)) & (nonzerox < (self.right_fit[0]*(nonzeroy**2) + 
                        self.right_fit[1]*nonzeroy + self.right_fit[2] + margin)))
        
        # Again, extract left and right line pixel positions
        self.left_x = nonzerox[left_lane_inds]
        self.left_y = nonzeroy[left_lane_inds] 
        self.right_x = nonzerox[right_lane_inds]
        self.right_y = nonzeroy[right_lane_inds]
        self.detected = True

        # evaluate polynomial
        self.evaluate_polynomial(left_lane_inds, right_lane_inds)

    
    def measure_curvature(self, ploty):
        '''
        Calculates the curvature of polynomial functions in meters.
        '''
        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30/720 # meters per pixel in y dimension
        xm_per_pix = 3.7/700 # meters per pixel in x dimension
        
        # Define y-value where we want radius of curvature
        # We'll choose the maximum y-value, corresponding to the bottom of the image
        y_eval = np.max(ploty)
        
        # Calculation of R_curve (radius of curvature pixel)
        self.left_rad_curve_pixel = ((1 + (2*self.left_fit[0]*y_eval + self.left_fit[1])**2)**1.5) / np.absolute(2*self.left_fit[0])
        self.right_rad_curve_pixel = ((1 + (2*self.right_fit[0]*y_eval + self.right_fit[1])**2)**1.5) / np.absolute(2*self.right_fit[0])

        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(self.left_y*ym_per_pix, self.left_x*xm_per_pix, 2)
        right_fit_cr = np.polyfit(self.right_y*ym_per_pix, self.right_x*xm_per_pix, 2)
        
        # Calculation of R_curve (radius of curvature metres)
        self.left_rad_curve_metre = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
        self.right_rad_curve_metre = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

        # Distance from center
        car_y = self.result.shape[0]
        left_line_location = self.left_fit[0]*car_y**2 + self.left_fit[1]*car_y + self.left_fit[2] 
        right_line_location = self.right_fit[0]*car_y**2 + self.right_fit[1]*car_y + self.right_fit[2]
        car_center = xm_per_pix*(self.result.shape[1] / 2)
        lane_center = xm_per_pix*((left_line_location + right_line_location) / 2)
        self.line_base_pos = car_center - lane_center

        # Add to image
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(self.result,'Left Curvature: %.2f m'%(self.left_rad_curve_metre),(60,60), font, 1,(255,255,255),2,cv2.LINE_AA)
        cv2.putText(self.result,'Right Curvature: %.2f m'%(self.right_rad_curve_metre),(60,100), font, 1,(255,255,255),2,cv2.LINE_AA)
        cv2.putText(self.result,'Distance from Center : %.2f m'%(self.line_base_pos),(60,140), font, 1,(255,255,255),2,cv2.LINE_AA)

        
    def evaluate_polynomial(self, left_lane_inds, right_lane_inds):
        # lane prediction confidence
        self.left_conf = 1. - abs(11000. - len(left_lane_inds))/ 15000.
        self.right_conf = 1. - abs(3000. - len(right_lane_inds))/ 8000.
        if len(left_lane_inds) < 11000:
            self.left_conf = 1.
        
        if self.left_conf < 0:
            self.left_conf = 0
            self.detected = False

        if len(right_lane_inds) < 3000:
            self.right_conf = 1.
            
        if self.right_conf < 0:
            self.right_conf = 0
            self.detected = False

        # check lane width
        if np.mean(self.right_x) - np.mean(self.left_x) > 1000 or np.mean(self.right_x) - np.mean(self.left_x) < 720:
            self.detected = False


        # check if pixels exist for each lane
        if self.right_x.size == 0 or self.right_y.size == 0:
            self.detected = False
            
        if self.left_x.size == 0 or self.left_y.size == 0:
            self.detected = False
