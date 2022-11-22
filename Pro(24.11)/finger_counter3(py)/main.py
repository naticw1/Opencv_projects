import cv2
import time
import HandTrackingModule as htm

############### Specifying the window size ###################
wCam, hCam = 640, 480
cap = cv2.VideoCapture(0)
cap.set(3, wCam) #width
cap.set(4, hCam) #Height



pTime = 0 #time to calculate FPS

detector = htm.handDetector(detectionCon=0.75)

tipIds = [4, 8, 12, 16, 20]

while True:
    success, img = cap.read() #read out frame
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    # print(lmList)

    #if the container is not empty
    if len(lmList) != 0:
        fingers = []

        # 4 Fingers
        for id in range(1, 5):
            if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        # Thumb
        # It is different for tamba because it bends differently than others

        if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)



        # print(fingers)
        totalFingers = fingers.count(1)
        print(totalFingers)


        cv2.rectangle(img, (20, 225), (170, 425), (0, 255, 0), cv2.FILLED) # print rectangle
        cv2.putText(img, str(totalFingers), (45, 375), cv2.FONT_HERSHEY_PLAIN, #put the number of fingers into the image
                    10, (255, 0, 0), 25)

    #calculate fps
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, f'FPS: {int(fps)}', (400, 70), cv2.FONT_HERSHEY_PLAIN, #put the number of fps into the image
                3, (255, 0, 0), 3)
    #show our frame
    cv2.imshow("Image", img)
    cv2.waitKey(1) #delay