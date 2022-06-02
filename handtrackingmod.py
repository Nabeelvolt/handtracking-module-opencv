import cv2

import mediapipe as mp
import time  # --> To check the frame rate


class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, modelComplexity=1, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.modelComplex = modelComplexity
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComplex,
                                        self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils  # it gives small dots on hands

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)  # --> Inbuilt

        # print(results.multi_hand_landmarks)  # --> Printing whether the hands are found or not

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:  # handLms means single hand

                if draw:
                    self.mpDraw.draw_landmarks(
                        img, handLms, self.mpHands.HAND_CONNECTIONS)

        return img

    def findPosition(self, img, handNo=0, draw=True):

        lmList = []  # --> This list will have all the landmarks position

        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]  # It will get the requered  hand and within that hand.. it will get the desired landmark

            for id, lm in enumerate(myHand.landmark):  # In this, the id number will relate to the exact index number of our finger landmark
                # Basically, Id number will be index number and it will be linked to the particular landmark
                # print(id,lm)

                h, w, c = img.shape  # --> Doing this to get the pixel value
                cx, cy = int(lm.x * w), int(lm.y * h)

                # print(id, cx, cy)

                lmList.append([id,cx,cy])

                # if (300<cx<330) and (200<cy<230):
                #     print("Opening app")

                if draw:
                    if id == 4:
                        cv2.circle(img, (cx, cy), 25, (255, 0, 255), cv2.FILLED)

        return lmList


def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)  # --> Video object
    detector = handDetector()

    while True:
        success, img = cap.read()
        frame = cv2.flip(img, 1)

        img = detector.findHands(frame)
        lmList=detector.findPosition(img)

        if len(lmList)!=0:
            print(lmList[4])   #--> It will give the land marks at index or we can say the landmarks of indice 4.. that is the thumb

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70),
                    cv2.FONT_HERSHEY_COMPLEX, 3, (255, 0, 255), 3)

        cv2.imshow("Image", img)

        if cv2.waitKey(1) & 0xFF == ord('d'):
            break


if __name__ == "__main__":
    main()