import os
import cv2
import numpy as np
directory = "./final_dataset-20221001T194638Z-001/final_dataset/Test_Final_2021_12_May/Weedelec/Bean/Images/"
save_directory = "./results"
count = 0
for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    image=cv2.imread(f)
    #cv2.imshow('input image',image)
    h, w, c = image.shape
    print(h,w)
    #cv2.waitKey(0)
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    edged=cv2.Canny(gray,200,300)# convert to LAB space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    # store the a-channel
    a_channel = lab[:,:,1]
    # Automate threshold using Otsu method
    th = cv2.threshold(a_channel,127,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]
    # Mask the result with the original image
    masked = cv2.bitwise_and(image, image, mask = th)
    cv2.imwrite(f'./results/weedelec{count}.jpg', masked)
    gray=cv2.cvtColor(masked,cv2.COLOR_BGR2GRAY)
    edged=cv2.Canny(gray,400,400)
    contours,hierarchy=cv2.findContours(edged,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
    cv2.imwrite(f'./results_edged/weedelec{count}.jpg', edged)
    cv2.drawContours(image,contours,-1,(0,255,0),3)
    cv2.imwrite(f'./results_contour/weedelec{count}.jpg', image)
    print(f"adding image {count}")
    count+= 1


# print('Numbers of contours found=' + str(len(contours)))
# cv2.drawContours(image,contours,-1,(0,255,0),3)
# cv2.imshow('contours',image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# print(filename)
# import cv2
# import numpy as np
# image=cv2.imread('./final_dataset-20221001T194638Z-001/final_dataset/Test_Final_2021_12_May/Weedelec/Bean/Images/weedelec_haricot1.jpg')
# cv2.imshow('input image',image)
# h, w, c = image.shape
# print(h,w)
# cv2.waitKey(0)
# gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
# edged=cv2.Canny(gray,400,400)
# cv2.imshow('canny edges',edged)
# cv2.waitKey(0)
# #cv2.imshow('canny edges after contouring', edged)
# #cv2.waitKey(0)
# contours,hierarchy=cv2.findContours(edged,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
# #print(contours)
# params = cv2.SimpleBlobDetector_Params()
# # params.filterByArea = 1
# # params.minArea = 50
# # params.maxArea = 10000000000
# # params.filterByColor = 1
# # params.blobColor = 0
# detector = cv2.SimpleBlobDetector_create(params)
# keypoints = detector.detect(edged)
# im_with_keypoints = cv2.drawKeypoints(edged, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# cv2.imshow("Keypoints", im_with_keypoints)
# cv2.waitKey(0)

# print('Numbers of contours found=' + str(len(contours)))
# cv2.drawContours(image,contours,-1,(0,255,0),3)
# cv2.imshow('contours',image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()