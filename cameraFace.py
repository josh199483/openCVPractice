import os, math, operator
from functools import reduce
import cv2
from PIL import Image

def createFace(facefile,msg,endstr):
    print(msg)
    camimage = cv2.VideoCapture(0)
    #確認鏡頭有開啟
    while(camimage.isOpened()):
        ret, img = camimage.read()
        if ret == True:
            cv2.imshow("test",img)
            #每0.1秒檢查一次有沒有按按鍵
            k = cv2.waitKey(100)
            if k == ord("z") or k == ord("Z"):
                cv2.imwrite(facefile,img)
                image = cv2.imread(facefile)
                face = faceCascade.detectMultiScale(image, 1.1, minNeighbors=5, minSize=(20, 20),
                                                    flags=cv2.CASCADE_SCALE_IMAGE)
                #只取第一張臉部圖片
                if not face:
                    #還是要轉換成同樣size的圖片以作比對
                    print("無法偵測到人臉")
                else:
                    (x,y,w,h) = (face[0][0],face[0][1],face[0][2],face[0][3])
                    image1 = Image.open(facefile).crop((x,y,x+w,y+h))
                    image1 = image1.resize((200,200),Image.ANTIALIAS)
                    image1.save(facefile)
                    break
    camimage.release()
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print(endstr)
faceCascade = cv2.CascadeClassifier('OpenCV/haarcascade/haarcascade_frontalface_default.xml')
recogimage = "recogimage.jpg"
loginimage = "loginimage.jpg"
if(os.path.exists(recogimage)):
    msg = "確認使用者是否可以登入\n按z比對"
    createFace(loginimage,msg,"")
    pic1 = Image.open(recogimage)
    pic2 = Image.open(loginimage)
    h1 = pic1.histogram()
    h2 = pic2.histogram()
    diff = math.sqrt(reduce(operator.add,list(map(lambda a,b:(a-b)**2,h1,h2)))/len(h1))
    if(diff<=100):
        print("pass,diff = {}".format(diff))
    else:
        print("error,diff = {}".format(diff))
else:
    msg = "建立使用者照片，按z拍照"
    endstr = "拍照完成"
    createFace(recogimage,msg,endstr)
