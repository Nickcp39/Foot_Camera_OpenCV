


img = imread('C:\01_Yanda_buffalo_2021Fall\research code\2022 0922 2D freehandheld system foot image\0909摄像头测试\20220909_test01\images\gray_image.png');

bbox = detectTextCRAFT(img,LinkThreshold=0.005);
Iout = insertShape(img,"rectangle",bbox,LineWidth=4);
figure
montage({img;Iout});
title("Input Image | Detected Text Regions")
bboxArea = bbox(:,3).*bbox(:,4);
[value,indx]= max(bboxArea);
roi = bbox(indx,:);
extractedImg = img(roi(2):roi(2)+roi(4),roi(1):roi(1)+roi(3),:);
figure
imshow(extractedImg)
title('Extracted Seven-Segment Text Region')


output = ocr(img,Language="seven-segment",TextLayout="word")
disp([output.Words])
Iocr = insertObjectAnnotation(img,"Rectangle",output.WordBoundingBoxes,output.Words,LineWidth=4,FontSize=20);
figure
imshow(Iocr)
