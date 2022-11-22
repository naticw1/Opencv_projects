import cv2
import numpy as np
import time

net = cv2.dnn.readNet('yolov3-tiny.weights', 'yolov3-tiny.cfg.txt') #network

classes = [] # list for storing the names of all recognized objects
with open("coco.names", "r") as file:

    # reaad file and store all class names
    classes = file.read().splitlines()

    print(classes)

cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_PLAIN
colors = np.random.uniform(0, 255, size=(100, 3))
fps = cap.get(cv2.CAP_PROP_FPS)
num_frames = 1
while True:
    _, img = cap.read()
    height, width, _ = img.shape

    start = time.time() # time for fps

    # convert the image into a blob
    blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), (0, 0, 0), swapRB=True, crop=False)

    net.setInput(blob)

    # get all output layers
    output_layers_names = net.getUnconnectedOutLayersNames()

    layerOutputs = net.forward(output_layers_names)

    boxes = []
    confidences = []
    class_ids = []

    #go to each of the layers (3 in total)
    for output in layerOutputs:

        #call each of the boxes as detection
        for detection in output:

            #remove first five values
            scores = detection[5:]
            # we find the index of the maximum value
            class_id = np.argmax(scores)

            #we get the exact value (maximum) according to the index
            confidence = scores[class_id]

            if confidence > 0.2:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                #we find the true length, width, recognized object (because there is a percentage in the list)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                #box showing the detected object
                boxes.append([x, y, w, h])

                #INFO about obj
                confidences.append((float(confidence)))
                class_ids.append(class_id)

    #To find out which of these boxes to save as an index
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.4)

    #loop through all indexes and show them
    if len(indexes) > 0:
        for i in indexes.flatten():

            #get the box coordinates of the object that was detected
            x, y, w, h = boxes[i]

            label = str(classes[class_ids[i]])
            confidence = str(round(confidences[i], 2))
            color = colors[i]

            # show a rectangle with an object
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

            # show name of object
            cv2.putText(img, label + " " + confidence, (x, y + 20), font, 2, (255, 255, 255), 2)

    # end time ( for fps counting )
    end = time.time()

    #fps calculation
    seconds = end - start
    fps = num_frames / seconds
    cv2.putText(img, "Fps : " + str(round(fps)), (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255))
    cv2.imshow('Image', img)

    #delay
    key = cv2.waitKey(1)

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
