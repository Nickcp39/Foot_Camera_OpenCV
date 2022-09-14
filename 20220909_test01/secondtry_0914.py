import cv2
import numpy as np

# 获取摄像头，传入0表示获取系统默认摄像头
cap = cv2.VideoCapture(cv2.CAP_DSHOW)
# 打开摄像头
cap.open(0)

while cap.isOpened():
    # 获取画面
    flag, frame = cap.read()

    if not flag:
        break

    # 获取键盘上按下哪个键
    key_pressed = cv2.waitKey(60)
    print("键盘上被按下的键是:" ,key_pressed)

    # 进行Canny边缘检测
    frame = cv2.Canny(frame, 100, 200)

    # 将单通道图像复制三份，摞成三通道图像
    frame = np.dstack((frame, frame, frame))

    # 在窗口显示
    cv2.imshow("my_window", frame)

    # 如果按下Esc键就退出循环
    if key_pressed == 27:
        break

# 关闭摄像头
cap.release()
# 关闭图像窗口
cv2.destroyAllWindows()