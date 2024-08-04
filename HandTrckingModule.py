import cv2
import mediapipe as mp
import time
import math


class handDetector:

    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands

        # self.hands = self.mpHands.Hands(self.mode, self.mpHands, self.detectionCon, self.trackCon)
        self.hands = self.mpHands.Hands()
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]  # 5个元素分别代表大拇指到小拇指的节号

    # 在输入的img中检测手部，将手部各关节节点标出并返回img
    def findHands(self, img, draw=True):  # 对传入的图像是否draw
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img)  # 对每一帧图像进行加工
        print(self.results.multi_hand_landmarks)  # 获取检测结果中的左右标签并打印

        if self.results.multi_hand_landmarks:  # 检测到手，并且返回标号
            for handLms in self.results.multi_hand_landmarks:  # 遍历所有手（maxHand）
                if draw:  # 是否标记出
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    # 得到21个手部节点在屏幕中的坐标
    def findPosition(self, img, handNo=0, draw=True):

        self.lmlist = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):  # 遍历枚举
                h, w, c = img.shape  # 获取画幅
                cx, cy = int(lm.x * w), int(lm.y * h)  # 比例放大，得到位置
                self.lmlist.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 3, (255, 0, 255), cv2.FILLED)
        return self.lmlist  # len(lmList) = 21, len(lmList[i]) = 3

    # 检测每个手指是否伸出
    def fingersUp(self):
        fingers = []
        # 大拇指
        if self.lmlist[self.tipIds[0]][1] > self.lmlist[self.tipIds[0] - 1][1] + 10:
            # 大拇指指尖节点x坐标 > 大拇指第二个节点的x坐标（+10） 则认为伸出大拇指
            fingers.append(1)
        else:
            fingers.append(0)

        # 其余手指
        for id in range(1, 5):
            if self.lmlist[self.tipIds[id]][2] < self.lmlist[self.tipIds[id] - 2][2]:
                # 其余四个手指指尖节点的y坐标 < 对应手指第三个节点的y坐标，则认为手指伸出
                fingers.append(1)
            else:
                fingers.append(0)

        # totalFingers = fingers.count(1)
        return fingers  # 5个元素对应从大拇指到小拇指 e.g. fingers = [0, 1, 0, 0, 0]代表只有食指伸出

    # 计算img中p1和p2节点之间的距离，返回距离length
    def findDistance(self, p1, p2, img, draw=True, r=15, t=3):
        x1, y1 = self.lmlist[p1][1:]
        x2, y2 = self.lmlist[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv2.circle(img, (x1, y1), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), r, (0, 0, 255), cv2.FILLED)
            length = math.hypot(x2 - x1, y2 - y1)  # p1和p2之间的欧几里得距离

        return length, img, [x1, y1, x2, y2, cx, cy]

# def main():
#     sum = 0
#     pTime = 0
#     cTime = 0
#     cap = cv2.VideoCapture(0)        # 打开摄像头，并初始化摄像头
#     detector = handDetector()
#
#     while True:
#         success, img = cap.read()     # 读取帧数并返回
#         img = detector.findHands(img)
#         lmlist = detector.findPosition(img)
#         fingers = detector.fingersUp(img)
#         length = detector.findDistance(img)
#         if len(lmlist) != 0:
#             print(lmlist[4])
#         for i in fingers:
#             sum += i
#         print(f'伸出{sum}个手指')
#         print(length)
#         cTime = time.time()
#         fps = 1 / (cTime - pTime)
#         pTime = cTime
#
#         cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
#
#         cv2.imshow("Take out your dog's paw", img)     # window 命名，并且显示
#         cv2.waitKey(1)    # 不断刷新图像帧率，每1ms刷新一次
#
# if __name__ =="__main__":
#     main()
