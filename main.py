import time

import cv2
import pandas as pd
from ultralytics import YOLO
import cvzone
import numpy as np
from tracker import *

model = YOLO('yolov8s.pt')


def MyProject(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        point = [x, y]
        #print(point)


cv2.namedWindow('MyProject')
cv2.setMouseCallback('MyProject', MyProject)

cap = cv2.VideoCapture('highwaytest4.mp4')  # video source

my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")
#print(class_list)

count = 0
countedcarup = 0
countedcardown = 0
countedtrckup = 0
countedtrckdown = 0
countedbikeup = 0
countedbikedown = 0
totalveh = 0
counted = 0


tracker = Tracker()
tracker1 = Tracker()
tracker2 = Tracker()
cy1 = 475
cy2 = 475
offset = 3

while True:
    ret, frame = cap.read()
    if not ret:
        break
    count += 1
    if count % 5 != 0:
        continue
    frame = cv2.resize(frame, (1280,720))

    results = model.predict(frame)
    # print(results)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")
    #print(px)

    list = []
    list1 = []
    list2 = []
    for index, row in px.iterrows():
        #print(row)


        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])
        c = class_list[d]
        trcka = class_list[d]
        cx3 = int(x1 + x2) // 2
        cy3 = int(y1 + y2) // 2
        cv2.circle(frame, (cx3, cy3), 4, (255, 0, 0), -1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
        cv2.putText(frame, str(c), (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), 1)
        if cy3 < (cy1 + offset) and cy3 > (cy1 - offset) and cx3 < 725:
            if c=='car':

              countedcardown=countedcardown+1
              counted=counted+1


            elif c=='truck':

               countedtrckdown = countedtrckdown + 1
               counted=counted+1


            elif c=='motorcycle':

                counted=counted+1


        if cy3 < (cy1 + offset) and cy3 > (cy1 - offset) and cx3 > 725:

            if c=='car':

                countedcarup = countedcarup + 1
                counted = counted + 1


            elif c=='truck':

                countedtrckup = countedtrckup + 1
                counted = counted + 1


            elif c == 'motorcycle':

                counted = counted + 1



        if 'car' in c:
            list.append([x1, y1, x2, y2])
            bbox_idx = tracker.update(list)

        elif 'motorcycle' in c:
            list1.append([x1, y1, x2, y2])
            bbox_idx1 = tracker1.update(list1)

        elif 'truck' in c:
            list2.append([x1, y1, x2, y2])
            bbox_idx2 = tracker2.update(list2)
        cv2.putText(frame, str('Downwards:'), (40, 90), cv2.FONT_HERSHEY_TRIPLEX, 0.6, (0, 0, 0), 1)
        cv2.putText(frame, str('CarDown:'), (20, 115), cv2.FONT_HERSHEY_TRIPLEX, 0.6, (0, 0, 0), 1)
        cv2.putText(frame, str('TruckDown:'), (20, 135), cv2.FONT_HERSHEY_TRIPLEX, 0.6, (0, 0, 0), 1)
        cv2.putText(frame, str(countedcardown), (120, 115), cv2.FONT_HERSHEY_TRIPLEX, 0.6, (0, 0, 0), 1)
        cv2.putText(frame, str(countedtrckdown), (140, 135), cv2.FONT_HERSHEY_TRIPLEX, 0.6, (0, 0, 0), 1)
        cv2.putText(frame, str('BikeDown:'), (20, 155), cv2.FONT_HERSHEY_TRIPLEX, 0.6, (0, 0, 0), 1)
        cv2.putText(frame, str(countedbikedown), (130, 155), cv2.FONT_HERSHEY_TRIPLEX, 0.6, (0, 0, 0), 1)

        cv2.putText(frame, str('Upwards:'), (1130, 90), cv2.FONT_HERSHEY_TRIPLEX, 0.6, (0, 0, 0), 1)
        cv2.putText(frame, str('COUNTEDTOTAL:'), (600, 90), cv2.FONT_HERSHEY_TRIPLEX, 0.6, (0, 0, 0), 1)
        cv2.putText(frame, str('CarUp:'), (1130, 115), cv2.FONT_HERSHEY_TRIPLEX, 0.6, (0, 0, 0), 1)
        cv2.putText(frame, str('TruckUp:'), (1130, 135), cv2.FONT_HERSHEY_TRIPLEX, 0.6, (0, 0, 0), 1)
        cv2.putText(frame, str(countedcarup), (1230, 115), cv2.FONT_HERSHEY_TRIPLEX, 0.6, (0, 0, 0), 1)
        cv2.putText(frame, str(countedtrckup), (1250, 135), cv2.FONT_HERSHEY_TRIPLEX, 0.6, (0, 0, 0), 1)
        cv2.putText(frame, str('BikeUp:'), (1130, 155), cv2.FONT_HERSHEY_TRIPLEX, 0.6, (0, 0, 0), 1)
        cv2.putText(frame, str(countedbikeup), (1230, 155), cv2.FONT_HERSHEY_TRIPLEX, 0.6, (0, 0, 0), 1)
        cv2.putText(frame, str(counted), (770, 90), cv2.FONT_HERSHEY_TRIPLEX, 0.8, (0, 0, 0), 1)







    cv2.line(frame, (225, cy1), (700, cy1), (0, 255, 0), 2)
    cv2.line(frame, (775, cy2), (1150, cy2), (0, 0, 255), 2)

    cv2.imshow("MyProject", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()