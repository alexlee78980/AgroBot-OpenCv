import cv2 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift, estimate_bandwidth
image=cv2.imread('./final_dataset-20221001T194638Z-001/final_dataset/Test_Final_2021_12_May/Weedelec/Bean/Images/weedelec_haricot1.jpg')
# convert to LAB space
lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
# store the a-channel
a_channel = lab[:,:,1]
# Automate threshold using Otsu method
th = cv2.threshold(a_channel,127,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]
# Mask the result with the original image
masked = cv2.bitwise_and(image, image, mask = th)
cv2.imshow('canny edgess',masked)
# # convert to RGB
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# pixel_values = image.reshape((-1, 3))
# # convert to float
# pixel_values = np.float32(pixel_values)
# criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
# # number of clusters (K)
# k = 2
# _, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
# # convert all pixels to the color of the centroids
# # convert back to 8 bit values
# centers = np.uint8(centers)

# # flatten the labels array
# labels = labels.flatten()
# segmented_image = centers[labels.flatten()]
# # reshape back to the original image dimension
# image = segmented_image.reshape(image.shape)
# # show the image
# # plt.imshow(segmented_image)
# # plt.show()
# light_green = (0,110,0)
# dark_green = (130,255,10)
# hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
# mask = cv2.inRange(hsv_image, light_green, dark_green)
# result = cv2.bitwise_and(image, image, mask=mask)
# plt.subplot(1, 2, 1)
# plt.imshow(mask, cmap="gray")
# plt.subplot(1, 2, 2)
# plt.imshow(result)
# plt.show()
# cv2.imshow('input image',image)
# h, w, c = image.shape
# print(h,w)
# cv2.waitKey(0)
gray=cv2.cvtColor(masked,cv2.COLOR_BGR2GRAY)
edged=cv2.Canny(gray,400,400)
cv2.imshow('canny edges',edged)
cv2.waitKey(0)
# cv2.imshow('canny edges after contouring', edged)
# cv2.waitKey(0)
contours,hierarchy=cv2.findContours(edged,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
print(contours)
# params = cv2.SimpleBlobDetector_Params()
# params.filterByArea = 1
# params.minArea = 50
# params.maxArea = 10000000000
# params.filterByColor = 1
# params.blobColor = 0
# detector = cv2.SimpleBlobDetector_create(params)
# keypoints = detector.detect(edged)
# im_with_keypoints = cv2.drawKeypoints(edged, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# cv2.imshow("Keypoints", im_with_keypoints)
# cv2.waitKey(0)

# print('Numbers of contours found=' + str(len(contours)))
# cv2.drawContours(image,contours,-1,(0,255,0),3)
# cv2.imshow('contours',image)
print('Numbers of contours found=' + str(len(contours)))
cv2.drawContours(image,contours,-1,(0,255,0),3)
cv2.imshow('contours',image)

cv2.waitKey(0)
cv2.destroyAllWindows()