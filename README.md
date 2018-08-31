# Neural-Best-Buddies / Cross Domain Feature Matching
Finding correspondences between a pair of images has been an important area of research due its numerous applications in image
processing and computer vision. Feature matching is a challenging problem as computer sees image as an array of numbers and even small
change in angle or illumination etc. changes array a lot making it seem a completely different image to the computer. A lot of progress has been made in finding correspondences in images of same / similar scene(s) with some variations. In these algorithms we extract features in image which are not much affected from the usual variations in images like illumination or occlusions etc. and hence can be matched. But what if we take an image of a swan and one of an airplane in different environments. Can we still find correspondences? Is the correspondence even well defined?

## Prerequisites
* Python 3.7 (lower versions also work correctly)
* PyTorch
* Skimage
* Torchvision
* Time
* Numpy
* Matplotlib
* PIL

## Getting Started
The main function in this project which user has to use is neural_best_buddies(). The functions takes in 2 images of any classes between which we have to find correspondences and outputs the found correspondences in each layer of interest.

## Arguments
img1 - location/name of the image 1 of shape 224 x 224 x 3 (as accepted by vgg19)

img2 - location/name of the image 2 of shape 224 x 224 x 3 (as accepted by vgg19)

## Return Values
lam - A list with 6 list elements. Each list element except 0th and the 5th represent the found matches in the corresponding layer. Final matches are in 1st list.

## Result
<img src="https://github.com/ayushgarg31/Neural-Best-Buddies/blob/master/images/test3.jpg" alt="drawing" height="224px" width="448px" style="float:left;"/> | <img src="https://github.com/ayushgarg31/Neural-Best-Buddies/blob/master/images/test1.jpg" alt="drawing"  height="224px" width="448px" style="float:left;"/> 
---------------------------------------------------------------|-------------------------------------------------------------------
<img src="https://github.com/ayushgarg31/Neural-Best-Buddies/blob/master/images/test2.jpg" alt="drawing"  height="224px" width="448px" style="float:left;"/> | <img src="https://github.com/ayushgarg31/Neural-Best-Buddies/blob/master/images/test4.jpg" alt="drawing"  height="224px" width="448px" style="float:left;"/>
