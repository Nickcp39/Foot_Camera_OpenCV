# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import cv2

def open_camera():
    # 找到设备 0
    cameraCapture_first = cv2.VideoCapture(0)
    cameraCapture_second = cv2.VideoCapture(1)

    cv2.namedWindow('Test camera')
    # 打开设备0
    cameraCapture_first.open(0)

    cameraCapture_second.open(1)


    #print("my frame parameter is {}".format(frame.shape))

    #print(frame.shape)
    if not cameraCapture_first.isOpened():
        print("I can't open it")
        return -1;
    """
    # 视频写入代码
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('lianzheng.avi', fourcc, 20.0, (640, 480))
    """

    while cameraCapture_first.isOpened():

        # 给o号设备设置名字
        success, frame = cameraCapture_first.read()
        success_02, frame_02 = cameraCapture_second.read()
        """
        
        等待用户指令， 如果用户点击esc， 就会关闭循环摄像头。
        waitKey() 函数的功能是不断刷新图像 , 频率时间为delay , 单位为ms  返回值为当前键盘按键值
        if cv2.waitKey(1) == 27:
        1和100 的区别就是让视频启动的时候等待多少秒再 show，等待用户其他指令
        """

        if cv2.waitKey(1) == 27:
            break

        cv2.imshow('Test camera', frame)
        #cv2.imshow("second Camera", frame_02)

    cameraCapture_first.release()
    cameraCapture_second.release()
    cv2.destroyAllWindows()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    open_camera()


