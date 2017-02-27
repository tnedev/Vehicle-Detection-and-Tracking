
**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run pipeline on a video stream,
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[hog]: ./output_images/hog_example1.png
[windows]: ./output_images/image5_windows.jpg
[heatmap]: ./output_images/heatmap.png
[boxes]: ./output_images/boxes.png

###Writeup / README

###Histogram of Oriented Gradients (HOG)

####1. Extract HOG Features

Extract HOG features is located in the utils.py under extract_hog_features function. 

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I explored the provided data and started experimenting with different parameters for the HOG feature extraction.
A typical result for this operation looks like this:

![alt text][hog]

####2. Choosing HOG Parameters

To settle on good parameters I did experiments with sample of 2000 training images to arrive with parameters which 
perform best in the test data set. 

At the end the fallowing was chosen: 
color_space='YCrCb'
spatial_size = (16, 16)
hist_bins = 32
hist_range = (0, 256)
orient = 12
pix_per_cell = 8
cell_per_block = 2
hog_channel = 'ALL'

####3. Training

At the end I used a feature vector combining both HOG features and color histogram. 
By doing so the algorithm was able to perform better and reduced the number of misclassified images.

The training pipeline is located in the utils.py under train function. 
The training data consisted of around 8500 labeled images for cars and notcars. 
The sets had a small difference in the count and I chose to exclude some of the images in order to have a full balanced 
training set. 

Because some images in the data set are sequential and obtained from video I chose to randomize their distribution and 
split the data in to training and test sets. 

The result was test accuracy of 99% on the test set. 

###Sliding Window Search

####  Sliding Window Search Implementation

Our training was implemented on 64x64x3 images each of which was either a closely cropped image of a car or not a car. 
In order to detect the cars on a full scale image containing the whole road we need to be able to chop the image into small 
sized windows (boxes) and check if the box is a car or not. 

Cars could appear large if they are close by or small if they are away. 
Therefore, we need to define both smaller and larger search windows.

An example search could look like this.

![alt text][image3]

To predict all those windows could take a while and do unnecessary calculation.
We could easily see that there is no point to search on the top half of the image because there are no flying cars on the market yet. 
We could also use the information that large windows will be close by and smaller will be away from the car. 

Our windows search is implemented in the find_cars function in detect.py straight on the pipeline implementation. 

After performing a sliding window search and throwing away all the boxes which we found no cars in we get
this image. 

![alt text][windows]

From this image we could see that the cars are correctly market, but there is also many false positives.
We could recognize two false positives. 

False positives from bad prediction.
False positives from overlapping windows. 

To remove the false positives we use a heatmap generation technique and averaging on a few frames in the video pipeline.


### Video Implementation

####1. Video Pipeline

Here's a ling of the final video result: https://www.youtube.com/watch?v=7mbs3X2dubc

The video processing is done in the VideoProcessor class in process_video.py

The video processor stores a few frames and implements a techniques to reduce the false positives. 
The lane detection stack was also added. 

####2. False Positives Filter

To remove the false positives and obtain a good tracking around vehicles we combine a few frames to produce
a vehicle detection heatmap. 

Many false positives will be detected on a single frame but won't be found on the next frame. Therefore, 
if we check for consistency over a few frames we could detect the false positives. 

This technique is implemented in the VideoProcessor.

The heatmap is generated over 3 frames and then only pixels above our threshold are taken into account. 

Here is how a heatmap looks. 
We could clearly see the 'hot' aread on the right and potential cold false detection on the left. 
![alt text][heatmap]

Finally, we draw a our boxes around the pixels which pass the threshold.

![alt text][boxes]

---

###Discussion

I am happy with the final result of the project. I see a lot of potential in this technique. 
I could also see some problem areas where I would like to work more and improve. 

Those include:
* False positives from the opposite lanes. In my video is clear that we detect cars driving on the opposite lanes. 
This is not necessary bad as this will be important if the cars were driving in both direction without separator. 
* It will be useful to only mask the polygon around the lanes as done in the lane detection algorithm. This will further improve on robustness. 
* We could see that the algorithm has problems separating between two cars while they are visually close to each other. It might be a good idea
to apply a filter which will compare the two positives for similarity before drawing the heatmap. 
* Tracking is good but in this algorithm we don't define separate lanes and distance from us. This will be the next step
in order for then build a control algorithm for the car. 

