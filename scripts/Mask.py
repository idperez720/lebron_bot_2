import cv2
from cv2 import IMREAD_GRAYSCALE
import numpy as np
import os
import time
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open webcam")
def Contrastador(imagen):

    # img = cv2.imread('imatext.png', IMREAD_GRAYSCALE)
    gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    imageArray=np.array(gray)
    fil= int(imageArray.shape[0])
    filmed = int(imageArray.shape[0]/2)
    col = int(imageArray.shape[1])
    
    columna=0
    flipedrArray=np.fliplr(imageArray)
    flipudArray=np.flipud(imageArray)
    for i, j in np.ndindex(imageArray.shape):
        if imageArray[i][j]>90:
            imageArray[i, j] = 255
        if imageArray[i][j]<90:
            imageArray[i, j] = 0
    
    for j in range(imageArray.shape[1]):
        
        if (imageArray[:,j]-imageArray[:,-j+3]).any()!=0:
            columna =j
            print(columna)
            break
    for j in range(imageArray.shape[1]):
        
        if (flipedrArray[:,j]-flipedrArray[:,j-3]).any()!=0:
            columna1 =j
            print(columna1)
            break

    for i in range(imageArray.shape[0]):
        
        if (imageArray[i,:]-imageArray[i-3,:]).any()!=0:
            fila =i
            print(fila)
            break
    for i in range(imageArray.shape[0]):
        
        if (flipudArray[i,:]-flipudArray[i-3,:]).any()!=0:
            fila1 =i
            print(fila1)
            break
    
    filaNew = int((fila)-20)
    fila1New = int((fila1)-20)
    rangefil=int(fil-fila1New)
    columnaNew=int(columna-20)
    columna1New=int(columna1-20)
    rangecol=int(col-columna1New)
    
    crop_img = imageArray[filaNew:rangefil, columnaNew:rangecol]
    kernel = np.ones((3, 3), np.uint8)
    imgMorph = cv2.erode(crop_img, kernel, iterations = 1)
    


       
    # lower_gray = np.array([0, 0, 0], np.uint8)
    # upper_gray = np.array([179, 50, 230], np.uint8)
    # mask_gray = cv2.inRange(imageArray, lower_gray, upper_gray)
    # img_res = cv2.bitwise_and(img, img, mask = mask_gray)
    cv2.imshow('Logo OpenCV',imgMorph)
    path = '/home/juan/Documents/MyCode/TextdetecTens/SimpleHTR/data'
    cv2.imwrite(os.path.join(path , 'word.png'), imgMorph)
    #cv2.imwrite('im.png', imageArray)
    
    t = cv2.waitKey(1)
k=0
while True:
    ret, frame = cap.read()
    if ret==False:break
    doc = frame
    cv2.imshow("Lector inteligente", frame)
    t = cv2.waitKey(1)
    if t==27:break
    

Contrastador(doc)
cap.release()
cv2.destroyAllWindows()

