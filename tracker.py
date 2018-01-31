import numpy as np
import cv2

cap = cv2.VideoCapture('DSC_0352.MOV')

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 29.97, (640,240))
frame = []
gray1 = []

if(cap.isOpened()):
    ret, frame1 = cap.read()

    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

while(cap.isOpened()):

    ret, frame2 = cap.read()

    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    diff = cv2.absdiff(gray1, gray2)

    thresh = cv2.threshold(diff, 50, 255, cv2.THRESH_BINARY)[1]
    
    edges = cv2.Canny(frame, 100, 200)

    final = cv2.addWeighted()

    cv2.imshow('Frame1', frame1)
    cv2.imshow('Frame2', frame2)
    cv2.imshow('Difference', thresh)

    out.write(diff)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()