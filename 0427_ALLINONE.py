import os
from PIL import Image
import pandas as pd
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
import openpyxl


import cv2
vidcap = cv2.VideoCapture('basicvideo_02.mp4')
success,image = vidcap.read()
count = 0
while success:
  cv2.imwrite("each_frames_new0426/frame%d.jpg" % count, image)     # save frame as JPEG file
  success,image = vidcap.read()
  print('Read a new frame: ', success)
  count += 1


DIGITS_LOOKUP = {
    (1, 1, 1, 0, 1, 1, 1): 0, #
    (0, 0, 0, 0, 1, 0, 0): 1, #
    (1, 0, 1, 1, 1, 0, 1): 2, #
    (1, 0, 1, 1, 0, 1, 1): 3, #
    (0, 1, 1, 1, 0, 1, 0): 4,  #
    (1, 1, 0, 1, 0, 1, 1): 5,  #
    (1, 1, 0, 1, 1, 1, 1): 6,  #
    (1, 0, 1, 0, 0, 1, 0): 7, #
    (1, 1, 1, 1, 1, 1, 1): 8, #
    (1, 1, 1, 1, 0, 1, 1): 9, #
    (0, 0, 0, 0, 0, 0, 0): '-'
}


def boundary_noise_removal_function(image,zip_list):
    x_total = 0;y_total = 0;w_total = 0;h_total = 0;

    for i in zip_list:
        (x,y,w,h) = i
        x_total = x+x_total; y_total = y+y_total;w_total = w+w_total;h_total = h+h_total;

    x_average = x_total/len(zip_list); y_average = y_total/len(zip_list);
    w_average = w_total/len(zip_list);    h_average = h_total/len(zip_list);

    for i in zip_list:
        (x,y,w,h) = i
        if(y<= (y_average*0.15)):
            zip_list.remove(i)
            print("i detect an abnormal region and remove it.")
        elif (w <= (w_average * 0.15)):
            zip_list.remove(i)
            print("i detect an abnormal region and remove it.")
        elif (h <= (h_average * 0.15)):
            zip_list.remove(i)
            print("i detect an abnormal region and remove it.")
    return(zip_list)

def float_point_removal_function(image,zip_list):
    #print("try to remove the float point",zip_list)
    # print("the number with float point position is",zip_list[3])
    (x,y,w,h) = zip_list[3]
    #print("xywh",x,y,w,h)
    image = cv2.rectangle(image, (x,y), (x+int(w*0.9),y+h), (125, 125, 0), 5)
    #cv2.imshow("float point",image)

    zip_list[3] = (x,y,int(w*0.9),h)
    return(zip_list)

def list_index_position(a):
    return sorted(range(len(a)), key=lambda k: a[k])

def number_selection(orginal_image,im_bw,zip_list):
    new_image = orginal_image
    output_image = im_bw
    whole_digits_list = []
    digit = 0
    w_total = 0; w_list = []
    for i in zip_list:
        x = i[0]
        y = i[1]
        w = i[2]
        h = i[3]
        w_list.append(w)
        x_half = int(x / 2); y_half = int(y / 2); w_half = int(w / 2); h_half = int(h/2);
        x_quater = int(x / 4); y_quater = int(y / 4); w_quater = int(w / 4); h_quater = int(h *0.20);
        #print( " the four is ", x, y, w, h)
        segments = [
            ((x+int(w*0.1), y), (x+int(w*0.9), h_quater+y), ("top section")),  # top
            ((x + int(w_quater * 0.35), y), (x + w_quater, y + h_half), "top left section"),  # top left
            ((x + w - w_quater, y), (x + w, y + h_half), "top right section"),  # top right
            ((x, y + int(h_half * 0.85)), (x + w, y + int(h_half * 1.15)), "middle section"),  # middle section
            ((x, y+h_half), (x+w_quater, y+h), ("bottom left section")),  # bottom left
            ((x+int(0.6*w),y+h_half), (x+int(0.9*w),y+h),"bottom right section"),  #bottom right
            ((x, y + int(h * 0.80)), (x + int(w*0.9), y + h), "bottom section"),  # bottom
        ]

        #((xA, yA), (xB, yB),(section_name)) = segments[6]
        #print(xA, yA, xB, yB,section_name, "the segments I try to draw is")
        roi = output_image

        on = [0] * len(segments)

        for (j, ((xA, yA), (xB, yB),section_name)) in enumerate(segments):
            # extract the segment ROI, count the total number of
            # thresholded pixels in the segment, and then compute
            # the area of the segment
            segROI = roi[yA:yB, xA:xB]
            total = cv2.countNonZero(segROI)
            area = (xB - xA) * (yB - yA)
            ROI_intensity = float(total / area)
            # if the total number of non-zero pixels is greater than
            # 50% of the area, mark the segment as "on"
            if(xA==256):
                print("ROI is",ROI_intensity)
            if ROI_intensity > 0.45:
                on[j] = 1
        try:
            digit = DIGITS_LOOKUP[tuple(on)]
        except:
            #print("some section number goes wrong")
            pass

        whole_digits_list.append(digit)


    average_width = int(sum(w_list) / len(zip_list) * 0.7)
    for i in range(len(w_list)):
        if w_list[i] < average_width:
            w_list[i] = average_width
            # print("i detect a number 1 ")
            whole_digits_list[i]=1

    # print(" the whole digits list is ",whole_digits_list)
    return whole_digits_list

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
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        # 去掉了一些边角位的杂音
        thresh = cv2.morphologyEx(thresh4, cv2.MORPH_OPEN, kernel)
        gray_image = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
        # res = cv2.matchTemplate(Edges, templateEdges, cv2.TM_CCORR)
        path_to_image = gray_image
        #print(text)
        # cv2.imshow("gray_image", final_image)
        (thresh, im_bw) = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        #cv2.imshow("im_bw", im_bw)

    # 处理好的黑白图像作为开始
    image_threshfilter = True;
    if image_threshfilter == True:
        kernel = np.ones((5, 5), dtype=np.uint8)
        im_dilate = cv2.dilate(im_bw, kernel, 3)

        # cv2.imshow("im_dilate", im_dilate)
        im_bw = im_dilate

        thresh = cv2.threshold(im_dilate, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 4))
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
    # image = thresh;
    for cnts in contours:
        #x, y, w, h = cv.boundingRect(cnts)
        #perimeter = cv.arcLength(cnts, True)
        #print(cnts)
        (x, y), radius = cv2.minEnclosingCircle(cnts)
        (x, y, radius) = np.int0((x, y, radius))  # 圆心和半径取整

        #cv2.circle(image, (x, y), radius, (0, 0, 255), 2)
        if radius >25 and radius <250:
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

    # cv2.imshow("im_bw",image)
    zip_list = list(zip(x_list, y_list,w_list,h_list))
    zip_list.sort()  # zip_list[0][0]
    Inital_zip_list = zip_list
    zip_list.sort(reverse=True) # zip_list[0][0]

    # print("the whole list is", zip_list)
    zip_list = boundary_noise_removal_function(image,zip_list)

    #print("list after noise removal is", zip_list)
    float_point_removal_function(image,zip_list)
    # print("after float point removal",zip_list)

    filter(None, zip_list)
    zip_list.sort()
    final_zip_list = zip_list
    output_individual_number_list = number_selection(image,im_bw, final_zip_list)

    num_list = output_individual_number_list
    num_str = ''.join(map(str, num_list))
    num_int = float(num_str)
    # print(num_int / 1000)
    output = (num_int / 1000)

    return output;

# Set the path to your image folder
image_folder = "x_frame/"
all_files = os.listdir(image_folder)
image_files = [filename for filename in all_files if filename.endswith(".png")]
num_images = len(image_files)

# Loop through the indices of the images you want to read
numbers = []
for i in range(1, num_images + 1):
    # Construct the filename for the current image index
    filename = f"x_frame_ROI_{i}.png"
    # Check if the image file exists in the folder
    if os.path.exists(os.path.join(image_folder, filename)):
        # Read the image using cv2.imread()
        img = cv2.imread(os.path.join(image_folder, filename))
        # Call the read_numbers() function to extract the number
        number = read_numbers(img)
        # Append the number to the list of numbers
        numbers.append(number)
        print(i,number)

# Save the list of numbers to a text file
np.savetxt("numbers_x.txt", numbers)

# Print a message to confirm that the file was saved
print(f"Numbers saved to numbers_x.txt")


# for Y
# Set the path to your image folder
image_folder = "y_frame/"
all_files = os.listdir(image_folder)
image_files = [filename for filename in all_files if filename.endswith(".png")]
num_images = len(image_files)

# Loop through the indices of the images you want to read
numbers = []
for i in range(1, num_images + 1):
    # Construct the filename for the current image index
    filename = f"y_frame_ROI_{i}.png"
    # Check if the image file exists in the folder
    if os.path.exists(os.path.join(image_folder, filename)):
        # Read the image using cv2.imread()
        img = cv2.imread(os.path.join(image_folder, filename))
        # Call the read_numbers() function to extract the number
        number = read_numbers(img)
        # Append the number to the list of numbers
        numbers.append(number)
        print(i,number)

# Save the list of numbers to a text file
np.savetxt("numbers_y.txt", numbers)

df = pd.DataFrame({"Numbers_y": numbers})

# Save the DataFrame to an Excel file
df.to_excel("numbers_y.xlsx", index=False)

# Print a message to confirm that the file was saved
print(f"Numbers saved to numbers_y.txt")