#**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/camera_calibration.png "Undistorted"
[image2]: ./output_images/undist_img.png "Road Transformed"
[image3]: ./output_images/img_pipeline.png "Threshold"
[image4]: ./output_images/warped.png "Warp Example"
[image5]: ./output_images/poly_fit.png "Fit Visual"
[image6]: ./output_images/final.png "Output"
[video1]: ./test_videos_output/project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points



### Camera Calibration

The code for this step is contained in the fourth code cell of the IPython notebook located in "./P2.ipynb".
I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]
A significant difference is observed at the image borders.

#### 2. Color and Gradient Thresholding

I used a combination of color and gradient thresholds to generate a binary image.

```python
# Convert to grayscale
gray = grayscale(undist)
#plt.imshow(gray, cmap='gray')

# X Thresholding
gradx = sobel_thresh(gray, orient='x', sobel_kernel=15, thresh=(20, 200))
#plt.imshow(gradx, cmap='gray')

# Y Thresholding
grady = sobel_thresh(gray, orient='y', sobel_kernel=15, thresh=(20, 200))
#plt.imshow(grady, cmap='gray')

# Magnitude Thresholding
mag_bin = mag_thresh(gray, sobel_kernel=15, thresh=(30, 200))
#plt.imshow(mag_bin, cmap='gray')

# Direction Thresholding
dir_bin = dir_thresh(gray, sobel_kernel=15, thresh=(0.7, 1.3))
#plt.imshow(dir_bin, cmap='gray')

# 'S' channel Thresholding
s_bin = hls_select(image, thresh=(150, 255))
#plt.imshow(s_bin, cmap='gray')

# 'R' channel Thresholding
r_bin = rgb_select(image, thresh=(150, 255))
#plt.imshow(r_bin, cmap='gray')
```
The final combination is as follows:

1. Pixels with high values for X and Y Thresholds along with high value for 'R' channel.
2. Pixels with high values for gradient magnitude and direction within 0.7 to 1.3 radians and high value for 'R' Channel
3. Pixels with high values for 'S' and 'R' channels.

These conditions are combined by 'OR' operators to produce the final binary.
```
# Combine thresholds
combination = np.zeros_like(dir_bin)
combination[(((gradx == 1) & (grady == 1) & (r_bin == 1)) | ((mag_bin == 1) & (dir_bin == 1) & (r_bin == 1)) | ((s_bin == 1) & (r_bin == 1)))] = 255
#plt.imshow(combination, cmap='gray')
```

Finally the image is masked so that unwanted lines are not detected. The masked region vertices and the warping *source* points are very close.

Here's an example of my output for this step.

![alt text][image3]

#### 3. Perspective Transform

The code for my perspective transform is as follows.
```python
# Tune coefficients to capture perspective
x1 = np.int(0.16 * img_size[0])
x2 = np.int(0.465 * img_size[0])
x3 = np.int(0.54 * img_size[0])
x4 = np.int(0.88 * img_size[0])
x5 = np.int(1 / 3 * img_size[0])
x6 = np.int(2 / 3 * img_size[0])

horizon = img_size[1]/1.6

src = np.float32([[x1, img_size[1]], [x2, horizon], [x3, horizon], [x4, img_size[1]]])
dst = np.float32([[x5, img_size[1]], [x5, 0], [x6, 0], [x6, img_size[1]]])

pts = np.array([[x1, img_size[1]], [x2, horizon], [x3, horizon], [x4, img_size[1]]], np.int32)
pts = pts.reshape((-1, 1, 2))

# Get perspective transform matrix
M = cv2.getPerspectiveTransform(src, dst)
Minv = cv2.getPerspectiveTransform(dst, src)
```
The perspective transform is calculated only once and hence no function has been implement. The same ``M`` and ``Minv`` matrices have been used to warp the videos.

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 205, 720      | 426, 720        | 
| 203, 720      | 595, 450      |
| 1127, 720     | 691, 450      |
| 695, 460      | 853, 720        |


I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Polynomial Fit

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

#### 5. Radius of Curvature and Offset

Following is the code for the curvature function


```python
def curvature(img_shape, left_fit, right_fit):
    '''
    Calculates the curvature of polynomial functions in meters.
    '''
    
    # Scaling factors in X and Y
    mx = 3 / 1000 # meters per pixel
    my = 30 / 720 # meters per pixel
    
    # Calculate radius at bottom most pixel
    y_eval = img_shape[1]
    y_cr = my * y_eval
    
    # Scale parabola
    scale = np.array([mx / my **2, mx / my, mx], np.float32)
    left_fit_cr = np.multiply(scale, left_fit)
    right_fit_cr = np.multiply(scale, right_fit)
    
    ##### TO-DO: Implement the calculation of R_curve (radius of curvature) #####
    left_curverad = ((1 + (2 * left_fit_cr[0] * y_cr + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit_cr[0])
    right_curverad = ((1 + (2 * right_fit_cr[0] * y_cr + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * right_fit_cr[0])
    
    # Calculate vehicle to lane center offset
    x_left = left_fit_cr[0] * y_cr ** 2 + left_fit_cr[1] * y_cr + left_fit_cr[2]
    x_right = right_fit_cr[0] * y_cr ** 2 + right_fit_cr[1] * y_cr + right_fit_cr[2]
    
    left_base_pos = x_left - img_shape[0] * mx / 2
    right_base_pos = x_right - img_shape[0] * mx / 2
    
    offset = ((x_left + x_right) - img_shape[0] * mx) / 2
    
    return left_curverad, right_curverad, offset, left_base_pos, right_base_pos
```

The X scaling factor has been calculated roughly basd on the lane width in the image (1000 pixels) and and assumed road width of 3 m.
The Y scaling factor has been calculated by assuming 30 m as the vertical distance in the warped image.

#### 6. Final Result

The final part is inverting the warp and stacking it over the original image. The following code does this step.

```python
# Create an image to draw the lines on
warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

# Recast the x and y points into usable format for cv2.fillPoly()
pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
pts = np.hstack((pts_left, pts_right))

# Draw the lane onto the warped blank image
cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

# Warp the blank back to original image space using inverse perspective matrix (Minv)
newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0])) 
# Combine the result with the original image
result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)

if(curv < 8000):
    text1 = 'Radius of Curvature- ' + str(round(curv, 2)) + ' meters'
else:
    curv = float('inf')
    text1 = 'Radius of Curvature- ' + str(round(curv, 2))

if(offset < 0):
    text2 = 'Vehicle ' + str(np.absolute(round(offset, 2))) + ' meters left of lane center'
else:
    text2 = 'Vehicle ' + str(round(offset, 2)) + ' meters right of lane center'

result = cv2.putText(result, text1, (img_size[0] // 30, img_size[1] // 15), cv2.FONT_HERSHEY_SIMPLEX, 0.75, [255, 255, 255], 2)
result = cv2.putText(result, text2, (img_size[0] // 30, img_size[1] // 9), cv2.FONT_HERSHEY_SIMPLEX, 0.75, [255, 255, 255], 2)

```

![alt text][image6]

---

### Pipeline (video)

Here's a [link to my video result](./test_videos_output/project_video.mp4)

---

### Discussion

The code works well for the project and the challenge video, but fails for the harder challenge video. The code mainly works because it averages out the lines over 20 frames, and hence it can estimate lanes even if the lane is not detected in the current frame.

This does not work for the harder challenge video as the road is very windy, and lanes are not visible for a major portion of the video. The algorithm works in glare and reflections, but cannot give good results for such an unpredicatable road.
