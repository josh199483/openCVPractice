from PIL import Image
import cv2
image1 = cv2.imread("OpenCV/connected_component_labeling_pictures/1.png")
image2 = cv2.imread("OpenCV/connected_component_labeling_pictures/2.png")

image3 = cv2.resize(image1,(640,480), interpolation = cv2.INTER_AREA)
print(image1.shape)
print(image2.shape)
print(image3.shape)
#dst=α*⋅img1+β*⋅img2+γ
dst = cv2.addWeighted(image3,0.7,image2,0.3,0)
cv2.imshow("dst",dst)

# I want to put logo on top-left corner, So I create a ROI
rows,cols,channels = image2.shape
roi = image3[0:rows, 0:cols ]
# Now create a mask of logo and create its inverse mask also
img2gray = cv2.cvtColor(image2,cv2.COLOR_BGR2GRAY)

ret, mask = cv2.threshold(img2gray, 160, 255, cv2.THRESH_BINARY)
cv2.imshow("mask",mask)
mask_inv = cv2.bitwise_not(mask)
cv2.imshow("mask_inv",mask_inv)
# Now black-out the area of logo in ROI
img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
# Take only region of logo from logo image.
img2_fg = cv2.bitwise_and(image2,image2,mask = mask)
# Put logo in ROI and modify the main image
dst = cv2.add(img1_bg,img2_fg)
image3[0:rows, 0:cols ] = dst
cv2.imshow('res',image3)
cv2.waitKey(0)
cv2.destroyAllWindows()
