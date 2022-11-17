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
import operator

DIGITS_LOOKUP = {
    (1, 1, 1, 1, 1, 1, 0): 0,
    (1, 1, 0, 0, 0, 0, 0): 1,
    (1, 0, 1, 1, 0, 1, 1): 2,
    (1, 1, 1, 0, 0, 1, 1): 3,
    (1, 1, 0, 0, 1, 0, 1): 4,
    (0, 1, 1, 0, 1, 1, 1): 5,
    (0, 1, 1, 1, 1, 1, 1): 6,
    (1, 1, 0, 0, 0, 1, 0): 7,
    (1, 1, 1, 1, 1, 1, 1): 8,
    (1, 1, 1, 0, 1, 1, 1): 9,
    (0, 0, 0, 0, 0, 1, 1): '-'
}


def list_index_position(a):
    return sorted(range(len(a)), key=lambda k: a[k])

def number_selection(orginal_image,im_bw,zip_list):
    new_image = orginal_image
    output_image = im_bw





    for i in zip_list:
        x = i[0]
        y = i[1]
        w = i[2]
        h = i[3]
        roi = output_image[y:y + h, x:x + w]

        (roiH, roiW) = roi.shape
        (dW, dH) = (int(roiW * 0.25), int(roiH * 0.15))
        dHC = int(roiH * 0.05)

        # define the set of 7 segments
        """
        segments = [
            ((0, 0), (w, dH)),  # top
            ((0, 0), (dW, h // 2)),  # top-left
            ((w - dW, 0), (w, h // 2)),  # top-right
            ((0, (h // 2) - dHC), (w, (h // 2) + dHC)),  # center
            ((0, h // 2), (dW, h)),  # bottom-left
            ((w - dW, h // 2), (w, h)),  # bottom-right
            ((0, h - dH), (w, h))  # bottom
        ]

        """
        segments = [
            ((x, y), (w, dH)),  # top
            ((x, y), (dW, h // 2)),  # top-left
            ((w - dW, y), (w, h // 2)),  # top-right
            ((x, (h // 2) - dHC), (w, (h // 2) + dHC)),  # center
            ((x, h // 2), (dW, h)),  # bottom-left
            ((w - dW, h // 2), (w, h)),  # bottom-right
            ((x, h - dH), (w, h))  # bottom
        ]

        on = [0] * len(segments)
        #print(on)
        for (i, ((xA, yA), (xB, yB))) in enumerate(segments):
            # extract the segment ROI, count the total number of
            # thresholded pixels in the segment, and then compute
            # the area of the segment
            segROI = roi[yA:yB, xA:xB]
            print(xA,xB, yA,yB)

            cv2.rectangle(new_image, (xA, yA), (xB, yB), (0, 0, 255), 3)

            total = cv2.countNonZero(segROI)
            area = (xB - xA) * (yB - yA)
            # if the total number of non-zero pixels is greater than
            # 50% of the area, mark the segment as "on"

            if total / float(area) > 0.4:
                on[i] = 1
            # lookup the digit and draw it on the image
    print(on)
    cv2.imshow("new_image", new_image)




def read_numbers(image):
    # pre-process the image by resizing it, converting it to
    # graycale, blurring it, and computing an edge map
    find_blackwhite = True
    if(find_blackwhite ==True):
        image = imutils.resize(image, height=500)
        ret, thresh4 = cv.threshold(image, 190, 255, cv.THRESH_TOZERO)
        image = thresh4
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        edged = cv2.Canny(blurred, 100, 200, 255)
        # thresh = cv2.threshold(thresh4, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        # 去掉了一些边角位的杂音
        thresh = cv2.morphologyEx(thresh4, cv2.MORPH_OPEN, kernel)
        gray_image = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
        # res = cv2.matchTemplate(Edges, templateEdges, cv2.TM_CCORR)
        path_to_image = gray_image
        path_to_tesseract = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        pytesseract.tesseract_cmd = path_to_tesseract
        final_image = thresh
        options = "outputbase digits"
        text = pytesseract.image_to_string(np.array(final_image))
        #print(text)
        # cv2.imshow("gray_image", final_image)
        (thresh, im_bw) = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        #cv2.imshow("im_bw", im_bw)

    # 处理好的黑白图像作为开始
    image_threshfilter = True;
    im_bw
    if image_threshfilter == True:
        kernel = np.ones((5, 5), dtype=np.uint8)
        im_dilate = cv2.dilate(im_bw, kernel, 3)
        cv2.imshow("im_dilate", im_dilate)
        im_bw = im_dilate
        thresh = cv2.threshold(im_dilate, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        contours, hierarchy = cv2.findContours(im_bw.astype(np.uint8).copy(), cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_SIMPLE)
        # contours, hierarchy = cv2.findContours(im_bw.astype(np.uint8).copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # print(len(contours))
        temp_index = 0
        index_list = []
        x_list = []
        y_list = []
        w_list = []
        h_list = []
        radius_list = []
    for cnts in contours:
        #x, y, w, h = cv.boundingRect(cnts)
        #perimeter = cv.arcLength(cnts, True)
        #print(cnts)


        (x, y), radius = cv2.minEnclosingCircle(cnts)
        (x, y, radius) = np.int0((x, y, radius))  # 圆心和半径取整
        if radius >25 and radius <200:
            #cv2.circle(image, (x, y), radius, (0, 0, 255), 2)
            x1, y1, w, h = cv.boundingRect(cnts)
            cv.rectangle(image, (x1, y1), (x1 + w, y1 + h), (255, 0, 0), 5)
            x_list.append(x1)
            y_list.append(y1)
            w_list.append(w)
            h_list.append(h)
            #print(x1,y1,w,h)
            index_list.append(temp_index)
            temp_index +=1
            #number_selection(im_bw,x1,y1,w,h)



    average_width = int(sum(w_list)/len(w_list)  * 0.7)
    for i in range (len(w_list)) :
        if w_list[i] < average_width:
            w_list[i] = average_width
            print("i detect a number 1 ")

    cv2.imshow("im_bw",image)
    zip_list = list(zip(x_list, y_list,w_list,h_list))
    zip_list.sort() # zip_list[0][0]
    print(zip_list)
    number_selection(image,im_bw, zip_list)
    # cv2.imshow("im_bw", image)
    cv2.waitKey(0)
    """
        
        for (i, ((xA, yA), (xB, yB))) in enumerate(segments):
            # extract the segment ROI, count the total number of
            # thresholded pixels in the segment, and then compute
            # the area of the segment
            segROI = roi[yA:yB, xA:xB]
            total = cv2.countNonZero(segROI)
            area = (xB - xA) * (yB - yA)
            # if the total number of non-zero pixels is greater than
            # 50% of the area, mark the segment as "on"

            if total / float(area) > 0.4:
                on[i] = 1
            # lookup the digit and draw it on the image

        digit = DIGITS_LOOKUP[tuple(on)]
        digits.append(digit)
        print(digits)
    


    
    """

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
    image = cv2.imread('images/77670.jpg')
    read_numbers(image)



