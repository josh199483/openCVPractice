import cv2
import numpy as np
from PIL import Image
import sys


#參數選擇0是以灰階呈現，預設是1
image = cv2.imread("d:/data/Downloads/1.jpg",1)
cv2.line(image,(30,30),(100,100),(255,0,0),2)
cv2.rectangle(image,(120,30),(180,60),(0,255,0),3)
cv2.circle(image,(50,80),30,(255,255,0),-1)
points = np.array([[130,120],[200,130],[170,180]],np.int32)
cv2.polylines(image,[points],True,(0,0,255),2)
cv2.imshow("test",image)
pic = sys.argv[1]
person = cv2.imread("OpenCV/face_pictures/{}.jpg".format(pic))
faceCascade = cv2.CascadeClassifier('OpenCV/haarcascade/haarcascade_frontalface_default.xml')
face = faceCascade.detectMultiScale(person,1.1,minNeighbors = 5,minSize = (10,10),flags = cv2.CASCADE_SCALE_IMAGE)
count = 1
image1 = Image.open("OpenCV/face_pictures/{}.jpg".format(pic))
for (x,y,w,h) in face:
    cv2.rectangle(person,(x,y),(x+w,y+h),(0,255,0),2)
    filename = "OpenCV/face_pictures/face{}.jpg".format(count)

    image2 = image1.crop((x,y,x+w,y+h))
    image3 = image2.resize((200,200),Image.ANTIALIAS)
    image3.save(filename)
    count+=1
cv2.imshow("face",person)
#save file,higher number means higher quality
#cv2.imwrite("d:/data/Downloads/1copy1.jpg",image,50)

#時間單位是毫秒，0表示無限長直到使用者按任意鍵才繼續
cv2.waitKey(0)
cv2.destroyAllWindows()
