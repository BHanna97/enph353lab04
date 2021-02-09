#!/usr/bin/env python

from PyQt5 import QtCore, QtGui, QtWidgets
from python_qt_binding import loadUi

import cv2 
import numpy as np
import sys  

class My_App(QtWidgets.QMainWindow):

	def __init__(self):

		super(My_App, self).__init__()
		loadUi("./SIFT_app.ui", self)
		#define camera ids. Found using v4l2 list devices.
		self._cam_id = 4 #cam 0 is external webcam
		self._cam_fps = 2 #frames per second for camera
		self._is_cam_enabled = False
		self._is_template_loaded = False

		self.browse_button.clicked.connect(self.SLOT_browse_button) #define functionality of browse button
		self.toggle_cam_button.clicked.connect(self.SLOT_toggle_camera) #define function for toggle_cam

		self._camera_device = cv2.VideoCapture(self._cam_id)#sets up cv2 camera object to capture video
		self._camera_device.set(3, 320) #set width to 320
		self._camera_device.set(4, 240) #set height to 240

		# Timer used to trigger the camera
		self._timer = QtCore.QTimer(self)
		self._timer.timeout.connect(self.SLOT_query_camera)
		self._timer.setInterval(100 / self._cam_fps) 

        #function defining browse button
        #when clicked, opens file path (curently SIFT_app directory)
	def SLOT_browse_button(self):
		dlg = QtWidgets.QFileDialog()
		dlg.setFileMode(QtWidgets.QFileDialog.ExistingFile)
		if dlg.exec_():
			self.template_path = dlg.selectedFiles()[0]

		pixmap = QtGui.QPixmap(self.template_path) #defines pixmap for image selected
		self.template_label.setPixmap(pixmap)
		print("Loaded template image file: " + self.template_path)
		


		#converts cv image to a pixmap
	def convert_cv_to_pixmap(self, cv_img):
		cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
		height, width, channel = cv_img.shape
		bytesPerLine = channel * width
		q_img = QtGui.QImage(cv_img.data, width, height, 
			bytesPerLine, QtGui.QImage.Format_RGB888)
		return QtGui.QPixmap.fromImage(q_img)


        #converts camera images to pixmap (ongoing). Captures a frame every timer interval

	def SLOT_query_camera(self):
		good_match = 10
		ret, img_train = self._camera_device.read() #saves return(boolean) and frame from camera
		
		img_query = cv2.imread(self.template_path, cv2.IMREAD_GRAYSCALE)
		gray_train = cv2.cvtColor(img_train, cv2.COLOR_BGR2GRAY) 

       	#create sift and find features
		sift = cv2.xfeatures2d.SIFT_create()

		kp_img, desc_img = sift.detectAndCompute(img_query, None)
		kp_gray, desc_gray = sift.detectAndCompute(gray_train, None)

       	#match features
		index_params = dict(algorithm = 0, trees = 5)
		search_params = dict()
       	#find matches for given params
		flann = cv2.FlannBasedMatcher(index_params, search_params)
		matches = flann.knnMatch(desc_img, desc_gray, k=2)

		good_points = []
		for m,n in matches:
			if m.distance < 0.8 * n.distance: #adjust for more precision
				good_points.append(m)

       	#applying RANSAC to sort inliers from outliers
       	#if there are not enough match points, just show the image
		if len(good_points) >= good_match:
			print("enough good points")
			query_points = np.float32([kp_img[m.queryIdx].pt for m in good_points]).reshape(-1, 1, 2)
			train_points = np.float32([kp_gray[m.trainIdx].pt for m in good_points]).reshape(-1, 1, 2)
			matrix, mask = cv2.findHomography(query_points, train_points, cv2.RANSAC, 3.0)
			match_mask = mask.ravel().tolist()
			h, w = img_query.shape
			points = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
			# print(points.shape)
			# print(matrix.shape)
			if matrix is not None:
				dst = cv2.perspectiveTransform(points, matrix)

				homography = cv2.polylines(img_train, [np.int32(dst)], True, (0, 255, 0), 3)
				pixmap = self.convert_cv_to_pixmap(homography)
				self.live_image_label.setPixmap(pixmap) #display the image
		else: #if there aren't enough good points, show possible matches
			print("draw matches only")
			print(len(kp_img))
			homography = cv2.drawMatches(img_query, kp_img, gray_train, kp_gray, good_points, img_train)		

			pixmap = self.convert_cv_to_pixmap(homography)
			self.live_image_label.setPixmap(pixmap) #display the image

        #turns the camera on and off and changes the label on the button
	def SLOT_toggle_camera(self):
		if self._is_cam_enabled:
			self._timer.stop()
			self._is_cam_enabled = False
			self.toggle_cam_button.setText("&Enable camera")
		else:
			self._timer.start()
			self._is_cam_enabled = True
			self.toggle_cam_button.setText("&Disable camera")



if __name__ == '__main__':
	app = QtWidgets.QApplication(sys.argv)
	myApp = My_App()
	myApp.show()
	print("in progress")
	sys.exit(app.exec_())