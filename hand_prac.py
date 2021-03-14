#This code is working, but little bit slow

import cv2, sys
import numpy as np
import imutils

cap = cv2.VideoCapture(0)

hand_cascade1 = cv2.CascadeClassifier(r'C:\Users\HP\cascade-models\handcascades\HandCascade.xml')
hand_cascade2 = cv2.CascadeClassifier(r'C:\Users\HP\cascade-models\handcascades\hand.xml')
hand_cascade3 = cv2.CascadeClassifier(r'C:\Users\HP\cascade-models\handcascades\palm.xml')
hand_cascade4 = cv2.CascadeClassifier(r'C:\Users\HP\cascade-models\handcascades\fist.xml')
#hand_cascade = cv2.CascadeClassifier(r'Hand_haar_cascade.xml')
while True:
    ret, img = cap.read()
    frame = imutils.resize(img, width = 600)
    frame = cv2.flip(frame, 1)
    clone = frame.copy()

    blur = cv2.GaussianBlur(frame,(5,5),0)
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    retval2,thresh1 = cv2.threshold(gray,70,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    
    #hand = hand_cascade.detectMultiScale(gray, 1.3, 7) #for HandCascade.xml
    hand1 = hand_cascade1.detectMultiScale(gray, 1.3, 7)
    hand2 = hand_cascade2.detectMultiScale(gray, 1.3, 7)
    hand3 = hand_cascade3.detectMultiScale(gray, 1.3, 7)
    hand4 = hand_cascade4.detectMultiScale(gray, 1.3, 7)
    #mask = np.zeros(thresh1.shape, dtype = "uint8")

    #for (x,y,w,h) in hand:
    for (x,y,w,h) in hand1:
        cv2.rectangle(frame,(x,y),(x+w,y+h), (0,255,0), 3)
        cv2.putText(frame,"Detected", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 3)
    for (x,y,w,h) in hand2:
        cv2.rectangle(frame,(x,y),(x+w,y+h), (0,255,0), 3)
        cv2.putText(frame,"Detected", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 3)
    for (x,y,w,h) in hand3:
        cv2.rectangle(frame,(x,y),(x+w,y+h), (0,255,0), 3)
        cv2.putText(frame,"Detected", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 3)
    for (x,y,w,h) in hand4:
        cv2.rectangle(frame,(x,y),(x+w,y+h), (0,255,0), 3)
        #cv2.rectangle(mask,(x,y),(x+w,y+h), (0,255,0), 3)
        cv2.putText(frame,"Detected", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 3)

    #cv2.imshow('Frame1', clone)
    #cv2.imshow('Frame2', blur)
    #cv2.imshow('Frame3', gray)
    cv2.imshow('Frame4', frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
