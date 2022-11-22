
import cv2
import pickle # using for saving all parking space boxes
import cvzone
import numpy as np

# Video feed
cap = cv2.VideoCapture('carPark.mp4')

#open file with all car positions
with open('CarParkPos', 'rb') as f:
    posList = pickle.load(f)

width, height = 107, 48


def checkParkingSpace(imgPro):
    spaceCounter = 0

    for pos in posList:

        #zoom in on the image to recognize the machine in the area
        x, y = pos
        imgCrop = imgPro[y:y + height, x:x + width]
        # cv2.imshow(str(x * y), imgCrop)

        #count of pixel( not black) in crop images
        count = cv2.countNonZero(imgCrop)

        # if count of pixels less 900 that it is free space( green) . Else there is car ( red)
        if count < 900:
            color = (0, 255, 0)
            thickness = 5
            spaceCounter += 1
        else:
            color = (0, 0, 255)
            thickness = 2


        #rectangle with pixel count
        cv2.rectangle(img, pos, (pos[0] + width, pos[1] + height), color, thickness)

        #count of free space
        cvzone.putTextRect(img, str(count), (x, y + height - 3), scale=1,
                           thickness=2, offset=0, colorR=color)

    cvzone.putTextRect(img, f'Free: {spaceCounter}/{len(posList)}', (100, 50), scale=3,
                       thickness=5, offset=20, colorR=(0, 200, 0))


while True:
    #CAP_PROP_POS_FRAMES - total amount of frames in video

    #CAP_PROP_FRAME_COUNT - current amount of frame

    if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):

        #reset frames so that the video is in a loop
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    #read the frame
    success, img = cap.read()

    #turn image in grey scale
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #blur image
    imgBlur = cv2.GaussianBlur(imgGray, (3, 3), 1)

    #convert into binary image
    imgThreshold = cv2.adaptiveThreshold(imgBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY_INV, 25, 16)

    #make the border width narrower for better recognition
    imgMedian = cv2.medianBlur(imgThreshold, 5)
    kernel = np.ones((3, 3), np.uint8)
    imgDilate = cv2.dilate(imgMedian, kernel, iterations=1)
    checkParkingSpace(imgDilate)
    cv2.imshow("Image", img)
    # cv2.imshow("ImageBlur", imgBlur)
    # cv2.imshow("ImageThres", imgMedian)
    key = cv2.waitKey(10)
    if key == ord('k'):
        break

#done