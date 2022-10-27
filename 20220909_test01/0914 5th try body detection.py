import cv2
import os
import numpy as np


# 所以global 在python中都 必须这么

font = cv2.FONT_HERSHEY_SIMPLEX  # 正常大小无衬线字体
size = 0.5
fx = 10
fy = 355
fh = 18
# ROI框的显示位置
x0 = 150
y0 = 100
# 录制的手势图片大小
width = 300
height = 300
# 每个手势录制的样本数
numofsamples = 300
counter = 0  # 计数器，记录已经录制多少图片了
# 存储地址和初始文件夹名称
gesturename = ''
path = ''

# 标识符 bool类型用来表示某些需要不断变化的状态
binaryMode = False  # 是否将ROI显示为而至二值模式
saveImg = False  # 是否需要保存图片


# 保存ROI图像
def saveROI(img):
    global path, counter, gesturename, saveImg
    if counter > numofsamples:
        # 恢复到初始值，以便后面继续录制手势
        saveImg = False
        gesturename = ''
        counter = 0
        return

    counter += 1
    name = gesturename + str(counter)  # 给录制的手势命名
    print("Saving img: ", name)
    cv2.imwrite(path + name + '.png', img)  # 写入文件
    time.sleep(0.05)


class camera_setting():

    def __init__( self, cap, camera_status,x0,y0):
        # 定义camera的两种功能
        self.cap = cap
        self.camera_status = camera_status
        self.x0 = x0
        self.y0 = y0
        self.cap = cv2.VideoCapture(0)  # 0为（笔记本）内置摄像头
        #self.success = success
        #self.raw_frame = raw_frame

    # 打开持续的摄像头
    def open_camera(self):

        success, raw_frame = self.cap.read()
        # print(type(success))
        # print(type(raw_frame))
        #cv2.imshow('Number Reading Camera', raw_frame)
        # 读帧
        ret, frame = self.cap.read()  # 返回的第一个参数为bool类型，用来表示是否读取到帧，如果为False说明已经读到最后一帧。frame为读取到的帧图片
        # 图像翻转（如果没有这一步，视频显示的刚好和我们左右对称）
        frame = cv2.flip(frame, 2)  # 第二个参数大于0：就表示是沿y轴翻转
        # 显示ROI区域 # 调用函数

        # 录制的手势图片大小
        width = 300
        height = 300
        # 录制的手势图片大小
        width_02 = 300
        height_02 = 100

        roi = camera_setting.binaryMask(frame, self.x0, self.y0, width,height)

        roi_02 = camera_setting.binaryMask(frame, self.x0, self.y0, width_02, height_02)

        # 显示提示语
        cv2.putText(frame, "Option: ", (fx, fy), font, size, (0, 255, 0))  # 标注字体
        cv2.putText(frame, "b-'Binary mode'/ r- 'RGB mode' ", (fx, fy + fh), font, size, (0, 255, 0))  # 标注字体
        cv2.putText(frame, "s-'new gestures(twice)'", (fx, fy + 2 * fh), font, size, (0, 255, 0))  # 标注字体
        cv2.putText(frame, "q-'quit'", (fx, fy + 3 * fh), font, size, (0, 255, 0))  # 标注字体

        #检查键盘输入
        camera_setting.keyboard_operation(self)

        # 展示处理之后的视频帧
        cv2.imshow('frame', frame)
        if (binaryMode):
            cv2.imshow('ROI', roi)
            cv2.imshow("ROI02",roi_02)
        else:
            cv2.imshow("ROI", frame[y0:y0 + height, x0:x0 + width])

    # 关闭摄像头
    def close_camera(self):
        # 最后记得释放捕捉
        self.cap.release()
        cv2.destroyAllWindows()
        print("I closed all windows as you need. see you next time!")

    # 键盘操作指令
    def keyboard_operation(self):

        key = cv2.waitKey(1) & 0xFF  # 等待键盘输入，
        if key == ord('b'):  # 将ROI显示为二值模式
            # binaryMode = not binaryMode
            binaryMode = True
            print("Binary Threshold filter active")
        elif key == ord('r'):  # RGB模式
            binaryMode = False
        elif key == ord('i'):  # 调整ROI框
            self.y0  = self.y0 - 5
        elif key == ord('k'):
            self.y0  = self.y0  + 5
        elif key == ord('j'):
            self.x0 = self.x0 - 5
        elif key == ord('l'):
            self.x0 = self.x0 + 5
        elif key == ord('q'):
            camera_setting.close_camera(self);
        elif key == ord('n'):
            # 开始录制新手势
            # 首先输入文件夹名字
            gesturename = (input("enter the gesture folder name: "))
            os.makedirs(gesturename)
            path = "./" + gesturename + "/"  # 生成文件夹的地址  用来存放录制的手势

    # 检查摄像头当前状态
    def check_camera_openornot(self):
        self.camera_status = self.cap.isOpened()  # 检测摄像头是否开启
        print(self.camera_status)
        return(self.camera_status)

    # 显示ROI为二值模式
    def binaryMask(frame, x0, y0, width, height):

        # 显示方框
        cv2.rectangle(frame, (x0, y0), (x0 + width, y0 + height), (0, 255, 0))
        half_height = int(height/2)
        half_width = int(width/2)
        #print(height, width, height/2, width/2)
        if(1):
            # 提取ROI像素 # 只针对在绿色框 rectangle中的图像， 边缘图像直接舍弃
            roi = frame[y0:y0 + height, x0:x0 + width]  # 0=>0.5
            # 高斯滤波处理
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 2)  # 高斯模糊，给出高斯模糊矩阵和标准差
            th3 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
            ret, res = cv2.threshold(th3, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)  # ret还是bool类型

            # 二值化处理 3,3
            kernel = np.ones((3, 3), np.uint8)  # 设置卷积核
            erosion = cv2.erode(res, kernel)  # 腐蚀操作 开运算：先腐蚀后膨胀，去除孤立的小点，毛刺
            # cv2.imshow("erosion", erosion)
            dilation = cv2.dilate(erosion, kernel)  # 膨胀操作 闭运算：先膨胀后腐蚀，填平小孔，弥合小裂缝
            # 轮廓提取
            binaryimg = cv2.Canny(res, 50, 200)  # 二值化，canny检测
            h = cv2.findContours(binaryimg, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)  # 寻找轮廓
            contours = h[0]  # 提取轮廓
            ret = np.ones(res.shape, np.uint8)  # 创建黑色幕布
            cv2.drawContours(ret, contours, -1, (255, 255, 255), 1)  # 绘制白色轮廓
            cv2.imshow("ret", ret)


def main():
    print("start the main program")
    #camera setting
    camera_status = False
    cap = None
    open_camera = True;
    camera_openornot=False;
    key = ord("p")
    #print(type(key))

    Camera01 = camera_setting(cap,camera_status, x0, y0)
    # 打开摄像头

    cap = Camera01.open_camera()
    #key = cv2.waitKey(1) & 0xFF

    while (open_camera == True):
        Camera01.open_camera()


if __name__ == '__main__':
    # 创建一个视频捕捉对象

    main()
