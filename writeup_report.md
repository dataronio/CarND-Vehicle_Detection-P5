# Vehicle Detection #

---

## **Vehicle Detection Project** ##


The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/plot1.png
[image2]: ./output_images/hog1.png
[image3]: ./output_images/hog2.png
[image4]: ./output_images/level1.png
[image5]: ./output_images/level2.png
[image6]: ./output_images/level3.png
[image7]: ./output_images/level4.png
[image8]: ./output_images/small_grid.png
[image9]: ./output_images/small_grid-2.png
[image10]: ./output_images/large_grid-1.png
[image11]: ./output_images/large_grid-2.png
[image12]: ./output_images/large_grid-3.png
[image13]: ./output_images/large_grid-4.png
[image14]: ./output_images/large_grid-5.png
[image15]: ./output_images/heatmaps.png
[video1]: ./project_video_output.mp4

---

### Files Submitted ###

My project includes the following files and directories:

* **CarND-Vehicle_Detection-P5.ipynb**  Jupyter notebook containing pipeline and all supporting functions.
* **/output_images**  Directory containing images processed and exported from notebook.
* **/VehicleData**  Directory containing images used for training and testing vehicle classifier.
* **/test_images**  Directory containing images for testing purposes.
* **project_video_output.mp4**  The project video processed by the vehicle detection pipeline.
* **writeup_report.md**  This report summarizing the results.

## Steps for Vehicle Detection Pipeline ##



### Histogram of Oriented Gradients (HOG) ###

I first visualized the training data and did a summary of class counts (car vs. not-car).  I found that the classes were reasonably balanced.  This makes the training of the classifier easier as there isn't a preponderance of one class over the other.  A visual of car and not-car examples follows.

![alt text][image1]

Most of my experimentation with the HOG features was completed during the project tutorial.  I settled on the following parameters rather quickly.

* **color_space =** 'HSV' - Hue, Saturation, and Value color space
* **orient =** 9  - number of HOG orientations
* **pix_per_cell =** 8  - HOG pixels per cell
* **cell_per_block =** 2  - HOG cells per block
* **hog_channel =** 2 - channel 2 is value
* **spatial_size =** (8, 8) - Spatial binning dimensions
* **hist_bins =** 64   - Number of histogram bins
* **spatial_feat =** True - Spatial features on 
* **hist_feat =** True - Histogram features on 
* **hog_feat =** True - HOG features on

These parameters are found in the seventh code block of the notebook.  

A visualization of the HOG features of a car image follows:

![alt text][image2]

While the HOG features of a non-car image follows:

![alt text][image3]

A data scaler is used for the features data. It is found in the eigth block of code as `X_scaler`. This transforms the data columnwise to be mean zero and unit variance. The image data is subject to a train test split of .80/.20 percent.  In the ninth block of code, a Support Vector machine with linear kernel is estimated on the training data.  The test accuracy is rather high at .9893.  I also estimate an ensemble of decision trees and get a test accuracy of .9797.  This is also rather good but I went with the linearSVC classifier in this project.

### Sliding Window Search ###

The basic principle of the sliding windows search is to use multiple scales of overlapping windows to detect vehicles on the road.  As the camera is front-facing on our car.  We are not likely to detect vehicles in the top half of the image.  Cars far from our car will by perspective look small and be clustered in a tight band around the middle of the image.

We can see an example of the finest scale grid in the image below:

![alt text][image4]

Examples of larger grids are given below:

![alt text][image5]
![alt text][image6]
![alt text][image7]

Overlapping larger grids lower in the image should detect vehicles that are closer to the car.

### Single Image Performance ###

First we test the small grids on the test images:

![alt text][image8]
![alt text][image9]

We can see multiple overlapping detections on the cars along with a few false positives.

Next lets test out larger grids on some test images:

![alt text][image10]
![alt text][image11]
![alt text][image12]
![alt text][image13]
![alt text][image14]

We can see that the detector appears to work alright.  We will need a method to deal with false positives and the multiple overlapping detections on the cars.

I deal with these issues with the heatmap device as implemented in lecture.  A minimum threshold parameter is used to help screen out false positives that only trigger a small number of grids.  The heatmap collects the positive grid detections and an overall maximum bounding box is estimated and displayed on the actual image.  This code is in `apply_threshold` and `draw_labeled_bboxes` in the code block labelled `Heatmap`.  A threshold paramenter of 3 is used and run on all test images in the following image.

![alt text][image15]







---

### Pipeline (video) ##

The final video output of the project video overall works but with some obvious issues.  There are intermitant false positives.  The white car is very difficult to detect after measures are taken to suppress false positives.  The black car, however, is very easily detected.  There is constant detection of the black car from the time it enters the frame to the end of the video.  At least there is one strightforward success story from the video.

Here's a [link to my video result](./project_video_output.mp4)

---

### Discussion ###

From analyzing the output of the project video, I have identified a number of weaknesses with this approach to vehicle detection.  Similar to the previous project on lane detection, it is not clear that there is a clear-cut method to crafting image features that would work perfectly in all conditions. 

Results show that even using a standard machine learning method such as Linear Support Vector Machines for classification there are serious issues with false positives.  Also, in the project video I was having a extremely difficult time in maintaining detection of the white car.  It is possible that using more training data with highly saturated white colored cars would help, I believe that the difficulties show the weaknesses of the traditional computer vision features with standard machine learning methods.

Just as there has been a revolution in image classification accuracy in the last 5 or so years, there have recently been great strides in various deep learning approaches to object detection.  Deep learning object detectors such as YOLO (You Only Look Once) and SSD (Single Shot Detector) have shown extremely accurate real-time object detection performance.  At best my approach can be run at say 4 to 5 frames per second while the newer deep learning methods can beat its accuracy and also run at true real time speeds of over 20 frames per second.  It does not seem reasonably to disregard such methods for advanced lane detection or vehicle detection.  While it is useful to learn the classical computer vision approaches, I believe a focus on deep learning methods is much needed for these difficult projects.
