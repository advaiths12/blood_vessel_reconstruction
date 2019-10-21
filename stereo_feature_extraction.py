import sys
import cv2
import math
import numpy as np

from matplotlib import pyplot as plt
from sklearn.preprocessing import normalize

l = 40000
sigma = 1.1
visual_mult = 1.0

class image_grabber:
	def __init__(self, path_to_video,rate):
		self.cap = cv2.VideoCapture(path_to_video)
		self.prev = None
		self.first = True
		print("created cap")

	def grab_and_send(self):
		print("in grab and send")
		if(not self.cap.isOpened()):
			print("Error in video file path")
		while(self.cap.isOpened()):

			ret, frame = self.cap.read()
			if(ret):
				split_width = frame.shape[1]//2
				left_img, right_img = cv2.GaussianBlur(frame[:, 0:split_width, :], (13, 13), 0), cv2.GaussianBlur(frame[:, split_width-1:-1, :], (13, 13), 0)
				left_img_alt = cv2.adaptiveThreshold(cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,  
	                                          cv2.THRESH_BINARY, 7, 3) 
				left_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
				right_img_alt = cv2.adaptiveThreshold(cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,  
	                                          cv2.THRESH_BINARY, 7, 3) 
				right_img = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
				# = frame[frame.shape[1]/2+1:][:]
				# print right_img.shape
				# print left_img.shape
				if(self.first):
					self.prev = left_img
					self.first = False
				else:
					cv2.imshow("Difference", left_img - self.prev)
				l_stereo = cv2.StereoSGBM_create(numDisparities=16, blockSize=3, 
											 uniquenessRatio=5, 
											 speckleWindowSize=0, 
											 speckleRange=2, 
											 preFilterCap=50, 
											 mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
				r_stereo = cv2.ximgproc.createRightMatcher(l_stereo)
				wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=l_stereo)
				wls_filter.setLambda(l)
				wls_filter.setSigmaColor(sigma)
				d_l = l_stereo.compute(left_img_alt, right_img)
				d_r = r_stereo.compute(right_img_alt, left_img)
				d_l = np.int16(d_l)
				d_r = np.int16(d_r)
				filtered = wls_filter.filter(d_l, left_img, None, d_r)
				filtered = cv2.normalize(src=filtered, dst=filtered, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)
				filtered = np.uint8(filtered)

				#disparity = cv2.convertScaleAbs(stereo.compute(left_img, right_img))
				cv2.namedWindow("Disparity", cv2.WINDOW_NORMAL)
				cv2.resizeWindow("Disparity", 800, 800)
				cv2.imshow("Left_alt", left_img_alt)
				cv2.imshow("Right_alt", right_img_alt)
				cv2.imshow("Unaltered", left_img)
				filtered = cv2.GaussianBlur(filtered, (3, 3), 0)
				cv2.imshow("Disparity", filtered)
	
				cv2.waitKey(50)
		cv2.destroyAllWindows()
		return 

def main():
	cv2.destroyAllWindows()
	# img = cv2.imread("/home/advaith/Desktop/ster.png")
	# split_width = img.shape[1]//2
	# left_img, right_img = img[:, 0:split_width, :], img[:, split_width:-1, :]
	# cv2.imshow("l", left_img)
	# cv2.imshow("r", right_img)
	# left_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
	# right_img = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
	# ster = cv2.StereoBM_create(numDisparities=32, blockSize=25)
	# disp = cv2.convertScaleAbs(ster.compute(left_img, right_img))
	# cv2.imshow("d", disp)
	ig = image_grabber("/home/advaith/Downloads/hamlyn_vids/stereo.avi", 24)
	ig.grab_and_send()
	ig.cap.release()
	cv2.destroyAllWindows()

main()