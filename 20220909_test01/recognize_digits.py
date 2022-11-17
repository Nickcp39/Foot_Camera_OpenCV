# -*- coding: utf-8 -*-
import cv2
from PIL import Image
import numpy as np
import time
from imutils.perspective import four_point_transform
from imutils import contours
import imutils
import cv2
import cv2 as cv
from PIL import Image
from pytesseract import pytesseract
from matplotlib import pyplot as plt
import argparse




def thermo_image_to_temp(image):

    # pre-process the image by resizing it, converting it to
    # graycale, blurring it, and computing an edge map
    image = imutils.resize(image, height=500)

    #127
    ret, thresh4 = cv.threshold(image, 190, 255, cv.THRESH_TOZERO)
    image = thresh4
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edged = cv2.Canny(blurred, 100, 200, 255)


    #thresh = cv2.threshold(thresh4, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    #去掉了一些边角位的杂音
    thresh = cv2.morphologyEx(thresh4, cv2.MORPH_OPEN, kernel)

    #cv2.imshow("thresh4", thresh4)
    #cv2.imshow("gray", gray)
    cv2.imshow("edged", edged)
    #cv2.imshow("thresh", thresh)

    gray_image  = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)

    #res = cv2.matchTemplate(Edges, templateEdges, cv2.TM_CCORR)
    path_to_image = gray_image
    path_to_tesseract = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    pytesseract.tesseract_cmd = path_to_tesseract

    final_image = thresh


    options = "outputbase digits"
    text = pytesseract.image_to_string(np.array(final_image) )
    print(text)
    cv2.imshow("gray_image", final_image)

    (thresh, im_bw) = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    cv2.imshow("im_bw", im_bw)



    # Filename
    filename = 'images/gray_image.png'
    #cv2.imwrite(filename, final_image)

    cv2.waitKey(0)

"""
    ret,thresh1 = cv.threshold(img,127,255,cv.THRESH_BINARY)
    ret,thresh2 = cv.threshold(img,127,255,cv.THRESH_BINARY_INV)
    ret,thresh3 = cv.threshold(img,127,255,cv.THRESH_TRUNC)
    #效果最好
    ret,thresh4 = cv.threshold(img,127,255,cv.THRESH_TOZERO)
    ret,thresh5 = cv.threshold(img,127,255,cv.THRESH_TOZERO_INV)
    titles = ['Original Image','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV']
    images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]
    
    for i in range(6):
        plt.subplot(2,3,i+1),plt.imshow(images[i],'gray',vmin=0,vmax=255)
        plt.title(titles[i])
        plt.xticks([]),plt.yticks([])
    plt.show()
"""




if __name__ == "__main__":

    image = cv2.imread('images/109725.png')

    thermo_image_to_temp(image)



