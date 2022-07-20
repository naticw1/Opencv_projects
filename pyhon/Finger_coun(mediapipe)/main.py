
# Idea : detect specific points in hand


import cv2
import time
import HandTrackingModule as htm



pastTime = 0
tipIds = [4,8,12,16,20]  # all id detection for finger counteing

count = 0

detector = htm.handDetector(detectionCon=0.75)  # Creates a basic Hand detector


video = cv2.VideoCapture(0)
while True:
    ret, img = video.read()


    img = detector.findHands(img)  # Detects the Hands
    points_list = detector.findPosition(img, draw=False)  # Trace landmarks or position


    if len(points_list) != 0:
        fingers = []

        #Action for thumb
        if points_list[tipIds[0]][1] > points_list[tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        # For another fingers
        for id in range(1, 5):
            if points_list[tipIds[id]][2] < points_list[tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        print(fingers)
        totalFingers = fingers.count(1)
        # print(totalFingers-1)
        cv2.putText(img, f"N. of fingers: {totalFingers}", (0, 50), cv2.FONT_HERSHEY_PLAIN,
                    3, (32,178,170), 3)

    currentTime = time.time()
    fps = 1 / (currentTime - pastTime)
    pastTime = currentTime

    cv2.putText(img, f" {int(fps)}", (580, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0),
                3)

    cv2.imshow("Camera", img)

    key_pressed = cv2.waitKey(1)
    if key_pressed == ord('q'):
        break

video.release()
cv2.destroyAllWindows()