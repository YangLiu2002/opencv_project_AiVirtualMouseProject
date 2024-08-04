import cv2
import numpy as np
import HandTrckingModule as htm
import time
# import autopy
import pyautogui as pg
###################
wCam, hCam = 640, 480   # 设置opencv窗口尺寸
frameR = 100
smoothening = 3
####################
cap = cv2.VideoCapture(0)  # 若笔记本自带摄像头参数为0，其他摄像头则1或者摄像头编号
cap.set(3, wCam)
cap.set(4, hCam)
pTime = 0
plocX, plocY = 0, 0
clocX, clocY = 0, 0

detector = htm.handDetector()
wScr, hScr = pg.size()  # 获取电脑尺寸，用于食指在cv窗口坐标和电脑坐标之间转换
# print(wScr, hScr)
while True:
    # 1.找到手部标识,得到关键点坐标
    success, img = cap.read()
    img = detector.findHands(img)   # 手的各个关节节点（共21个）在img中被标出
    # 画绿色框
    cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - int(frameR*1.8)), (0, 255, 0), 2, cv2.FONT_HERSHEY_PLAIN)
    # len(lmlist)=21, len(lmlist[i])=3,e.g. lmlist[8]=[8, xx, yy]代表食指指尖8号在cv窗口的坐标为（xx, yy）
    lmlist = detector.findPosition(img, draw=True)

    # 2.判断食指和中指是否伸出
    if len(lmlist) != 0:
        x1, y1 = lmlist[8][1:]   # 食指指尖在cv窗口中的坐标
        x2, y2 = lmlist[12][1:]   # 中指指尖在cv窗口中的坐标
        fingers = detector.fingersUp()  # 判断手指是否伸出
        # print(fingers)

        # 3.若只有食指伸出，则进入移动模式
        if fingers[1] and fingers[0] == False and fingers[2] == False and fingers[3] == False and fingers[4] == False:
            # 4.坐标转换：将食指在窗口的坐标转换（等比缩放）为鼠标在桌面上的坐标
            # 得到鼠标坐标（mouse_x, mouse_y） [线性插值函数 val = np.interp(x, arr1, arr2);arr1为横坐标值，arr2为纵坐标值，根据arr1和arr2将x映射到val]
            mouse_x = np.interp(x1, (frameR, wCam - frameR), (0, wScr))   # 将上方画的绿色框的宽度映射到屏幕宽度，将窗口横坐标x1映射为屏幕横坐标mouse_x
            mouse_y = np.interp(y1, (frameR, hCam- int(frameR*1.8)), (0, hScr))   # 绿色框的高度->屏幕高度，窗口纵坐标y1->屏幕横坐标mouse_y

            # smoothening valuesq
            clocX = plocX + (mouse_x - plocX) / smoothening
            clocY = plocY + (mouse_y - plocY) / smoothening
            print(wScr - clocX, clocY)
            # wScr - clocX为了在横向方向上和手指移动方向镜像； 1e-6的目的是防止位置超出边界ValueError：Point out of bounds
            pg.moveTo(clocX, clocY)

            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)   # 在食指指尖部位画圆

            plocX, plocY = clocX, clocY    # 更新前一帧的鼠标所在位置坐标，将当前帧鼠标所在位置，变成下一帧的鼠标前一帧所在位置

        # 5.若食指和中指伸出，检测指头距离，距离够短则对应鼠标点击
        if fingers[1] and fingers[2] and fingers[0] == False and fingers[3] == False and fingers[4] == False:
            length, img, pointInfo = detector.findDistance(8, 12, img)   # 8节点是食指指尖，12节点是中指指尖
            if length < 40:
                cv2.circle(img, (pointInfo[4], pointInfo[5]), 15, (0, 255, 0), cv2.FILLED)
                pg.leftClick()

        # 6.若只有食指和大拇指伸出，则鼠标滚轮向下翻滚
        if fingers[0] and fingers[1] and fingers[2] == False and fingers[3] == False and fingers[4] == False:
            cv2.circle(img, (lmlist[4][1], lmlist[4][2]), 15, (0, 255, 0), cv2.FILLED)  # 大拇指指尖绘圆
            cv2.circle(img, (lmlist[8][1], lmlist[8][2]), 15, (255, 0, 255), cv2.FILLED)   # 食指指尖画圆
            pg.scroll(-100)  # 正值向上，负值向下

        # 7.若食指和小拇指伸出，则鼠标滚轮向上翻滚
        if fingers[1] and fingers[4] and fingers[0] == False and fingers[2] == False and fingers[3] == False:
            cv2.circle(img, (lmlist[20][1], lmlist[20][2]), 15, (0, 255, 0), cv2.FILLED)  # 小拇指指尖绘圆
            cv2.circle(img, (lmlist[8][1], lmlist[8][2]), 15, (255, 0, 255), cv2.FILLED)  # 食指指尖画圆
            pg.scroll(100)  # 正值向上，负值向下

        # 8.四指伸出大拇指不伸出，则点击鼠标右键
        if fingers[0] == False and fingers[1] and fingers[2] and fingers[3] and fingers[4] == False:
            cv2.circle(img, (lmlist[8][1], lmlist[8][2]), 15, (0, 255, 0), cv2.FILLED)
            cv2.circle(img, (lmlist[12][1], lmlist[12][2]), 15, (0, 255, 0), cv2.FILLED)
            cv2.circle(img, (lmlist[16][1], lmlist[16][2]), 15, (0, 255, 0), cv2.FILLED)
            cv2.circle(img, (lmlist[20][1], lmlist[20][2]), 15, (0, 255, 0), cv2.FILLED)
            pg.rightClick()
            pg.press('enter')







    # 11.检查是否获得足够的帧率
    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    # 12.显示
    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()






























