# This code is working

import cv2, sys
import numpy as np
import imutils

cap = cv2.VideoCapture(0)

#hand_cascade = cv2.CascadeClassifier(r'C:\Users\HP\cascade-models\handcascades\.xml')
hand_cascade = cv2.CascadeClassifier(r'Hand_haar_cascade.xml')
while True:
    ret, img = cap.read()
    frame = imutils.resize(img, width = 600)
    frame = cv2.flip(frame, 1)
    clone = frame.copy()

    blur = cv2.GaussianBlur(frame,(5,5),0)
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    #hand = hand_cascade.detectMultiScale(gray, 1.3, 7) #for Hand.Cascade.xml
    hand = hand_cascade.detectMultiScale(gray, 1.3, 7)

    for (x,y,w,h) in hand:
        cv2.rectangle(frame,(x,y),(x+w,y+h), (0,255,0), 2)
        cv2.putText(frame,"Detected", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 2)

    cv2.imshow('Frame1', clone)
    cv2.imshow('Frame2', frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
        
#sys.exit()
