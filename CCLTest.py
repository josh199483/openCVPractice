import numpy as np
import cv2
from skimage import measure
from imutils import contours
import imutils
#參數選擇0是以灰階呈現，預設是1
image = cv2.imread("OpenCV/connected_component_labeling_pictures/3.jpg")

#可看到各channels的在圖片上矩陣的數值
b,g,r = cv2.split(image)
img = cv2.merge((b,g,r))
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
#作糊化時，需指定核心大小，核心越大模糊效果越高
# blur = cv2.medianBlur(gray,5)
blur = cv2.GaussianBlur(gray, (11, 11), 0)
# V = cv2.split(cv2.cvtColor(blur, cv2.COLOR_BGR2HSV))[2]
# thresh = threshold_adaptive(V, 47, offset=15).astype("uint8") * 255
cv2.imshow("blur",blur)
#必須先做灰階才可做threshold
# thresh = cv2.adaptiveThreshold(blur,47,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,blockSize=9,C=3)
# thresh = cv2.bitwise_not(thresh)
thresh = cv2.threshold(blur, 200, 255, cv2.THRESH_BINARY)[1]
print(thresh.shape)
cv2.imshow("thresh",thresh)
# perform a series of erosions and dilations to remove
# any small blobs of noise from the thresholded image
thresh = cv2.erode(thresh, None, iterations=2)
thresh = cv2.dilate(thresh, None, iterations=4)
cv2.imshow("thresh1",thresh)
#image blending，兩張圖片需要是同一種channel(也就是一起灰階或一起彩色)
dst = cv2.addWeighted(gray,0.7,thresh,0.3,0)
# cv2.imshow("ds",dst)
labels = measure.label(thresh, neighbors=8, background=0)
mask = np.zeros(thresh.shape, dtype="uint8")
print("[INFO] Total {} blobs".format(len(np.unique(labels))))
print(len(labels))
print(np.unique(labels))
#依序處理每個labels
for (i, label) in enumerate(np.unique(labels)):
#如果label=0，表示它為背景
    if label == 0:
        print("[INFO] label: 0 (background)")
        continue
#否則為前景，顯示其label編號l
    print("[INFO] label: {} (foreground)".format(i))
#建立該前景的Binary圖
    labelMask = np.zeros(thresh.shape, dtype="uint8")
    labelMask[labels == label] = 255
#有幾個非0的像素?
    numPixels = cv2.countNonZero(labelMask)
#如果像素數目在2500~4000之間認定為車牌字母或數字
    if numPixels > 300:
#放到剛剛建立的空圖中
        mask = cv2.add(mask, labelMask)
cv2.imshow("label",mask)
# find the contours in the mask, then sort them from left to
# right
cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]
cnts = contours.sort_contours(cnts)[0]
# loop over the contours
for (i, c) in enumerate(cnts):
# draw the bright spot on the image
	(x, y, w, h) = cv2.boundingRect(c)
	((cX, cY), radius) = cv2.minEnclosingCircle(c)
	cv2.circle(image, (int(cX), int(cY)), int(radius),(0, 0, 255), 3)
	cv2.putText(image, "#{}".format(i + 1), (x, y - 15),cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
cv2.imshow("Image", image)
cv2.waitKey(0)