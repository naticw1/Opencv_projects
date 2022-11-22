import cv2
import pickle

#width, height of car
width, height = 107, 48

try:
    # CarParkPos is a prepared file with all positions of cars
    with open('CarParkPos', 'rb') as f:
        posList = pickle.load(f)
except:
    posList = []

#a function for recognizing a click on an image
def mouseClick(events, x, y, flags, params):

    #left button to add object
    if events == cv2.EVENT_LBUTTONDOWN:
        posList.append((x, y))

    #right button to delete the object
    if events == cv2.EVENT_RBUTTONDOWN:
        for i, pos in enumerate(posList):
            x1, y1 = pos
            if x1 < x < x1 + width and y1 < y < y1 + height:
                posList.pop(i)
    # rewrite the information about the boxes to the file (if there were no changes, the file will be the same)
    with open('CarParkPos', 'wb') as f:
        pickle.dump(posList, f)


while True:

    # read image ( that image is from the same camera as the video)
    img = cv2.imread('carParkImg.png')

    #put rectangle for each positions in the list
    for pos in posList:
        cv2.rectangle(img, pos, (pos[0] + width, pos[1] + height), (255, 0, 255), 2)

    #default things
    cv2.imshow("Image", img)
    cv2.setMouseCallback("Image", mouseClick)
    cv2.waitKey(1)