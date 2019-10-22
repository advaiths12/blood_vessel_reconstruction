import sys
import cv2
import math
import numpy as np
from sklearn.preprocessing import normalize
import open3d as o3d

l = 40000
sigma = 1.1
visual_mult = 1.0

class image_grabber:
	def __init__(self, path_to_video,rate):
		self.cap = cv2.VideoCapture(path_to_video)
		self.prev = None
		self.first = True
		self.viz = o3d.visualization.Visualizer()
		self.viz.create_window()
		print("created cap")

	def grab_and_send(self):
		print("in grab and send")
		if(not self.cap.isOpened()):
			print("Error in video file path")
		pcd = o3d.geometry.PointCloud()
		while(self.cap.isOpened()):

			ret, frame = self.cap.read()
			if(ret):

				pcd.clear()
				split_width = frame.shape[1]//2
				left_img, right_img = frame[:, 0:split_width, :], frame[:, split_width-1:-1, :]
				rgb_img = left_img
				left_img_alt = cv2.adaptiveThreshold(cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
	                                          cv2.THRESH_BINARY, 7, 3)
				left_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
				right_img_alt = cv2.adaptiveThreshold(cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
	                                          cv2.THRESH_BINARY, 7, 3)
				right_img = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
				# = frame[frame.shape[1]/2+1:][:]
				# print right_img.shape
				# print left_img.shape

				l_stereo = cv2.StereoSGBM_create(numDisparities=16, blockSize=3,
											 uniquenessRatio=20,
											 speckleWindowSize=0,
											 speckleRange=2,
											 preFilterCap=20,
											 mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
				r_stereo = cv2.ximgproc.createRightMatcher(l_stereo)
				wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=l_stereo)
				wls_filter.setLambda(l)
				wls_filter.setSigmaColor(sigma)
				d_l = l_stereo.compute(left_img, right_img)
				d_r = r_stereo.compute(right_img, left_img)
				d_l = np.int16(d_l)
				d_r = np.int16(d_r)
				filtered = wls_filter.filter(d_l, left_img, None, d_r)
				filtered = cv2.normalize(src=filtered, dst=filtered, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)
				filtered = np.uint8(filtered)
				#disparity = cv2.convertScaleAbs(stereo.compute(left_img, right_img))
				rgb_img1 = o3d.geometry.Image(rgb_img)
				filtered_o3d = o3d.geometry.Image(filtered)
				rgbd_img = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb_img1, filtered_o3d)

				pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_img, o3d.camera.PinholeCameraIntrinsic(
            o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))
				# flip_transform = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
				# pcd.transform(flip_transform)
				self.viz.add_geometry(pcd)
				self.viz.update_geometry()
				self.viz.poll_events()
				self.viz.update_renderer()
				cv2.namedWindow("Disparity", cv2.WINDOW_NORMAL)
				cv2.resizeWindow("Disparity", 800, 800)
				cv2.imshow("Left_alt", left_img_alt)
				cv2.imshow("Right_alt", right_img_alt)
				cv2.imshow("Unaltered", rgb_img)
				cv2.imshow("Disparity", filtered)
				#o3d.visualization.draw_geometries([pcd])

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
	ig = image_grabber("./stereo1.avi", 24)
	ig.grab_and_send()
	ig.cap.release()
	cv2.destroyAllWindows()

main()
