# blood_vessel_reconstruction
Creating disparity and point clouds based on stereo endoscopy videos from robotic surgery 

Evaluated on the Hamlyn Dataset

1. Uses Stereo Block Matching algorithms to estimate disparity map
2. Uses data from images passed through a Gaussian Adaptive Threshold to isolate blood vessels
3. Cross matches blocks from left and right pairs alternating with blood vessel-thesholded images
4. Produces a real time disparity map of the surgery site

Dependencies:
OpenCV Contrib
Numpy
Sklearn

![alt text](https://github.com/advaiths12/blood_vessel_reconstruction/blob/master/doc_1.png)
