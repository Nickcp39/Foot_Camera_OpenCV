import cv2
import os
cap= cv2.VideoCapture(0)

width= int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height= int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

print(fps, width, height, "fps, width, height")

base_filename = "basicvideo.mp4"
filename, file_extension = os.path.splitext(base_filename)
count = 1

while os.path.exists(base_filename):
    base_filename = f"{filename}_{count:02d}{file_extension}"
    count += 1


writer= cv2.VideoWriter(base_filename, cv2.VideoWriter_fourcc(*'DIVX'), fps, (width,height))

recording = False
while True:
    keyPress = cv2.waitKey(1)
    ret,frame= cap.read()
    frame = cv2.flip(frame, 0)
    frame = cv2.flip(frame, 1)

    if(recording == True):
        writer.write(frame)

    cv2.imshow('frame', frame)

    if keyPress == 27:
        break

    if keyPress & 0xFF == ord('s'):
        recording = True
        print("start recording")

    elif keyPress & 0xFF == ord('n'):
        recording = False
        print("stop recording")
        break

cap.release()
writer.release()
cv2.destroyAllWindows()




