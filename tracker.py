import numpy as np
import cv2

cap = cv2.VideoCapture('DSC_0353.MOV')

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 29.97, (640,240))
frame = []
gray1 = []

if(cap.isOpened()):
    ret, frame1 = cap.read()

    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

while(cap.isOpened()):

    # 1) DIFFERENCE DETECTION
    ret, frame2 = cap.read()

    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    diff = cv2.absdiff(gray1, gray2)

    thresh = cv2.threshold(diff, 60, 255, cv2.THRESH_BINARY)[1]

    _, contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print(len(contours))
    contourimg = cv2.drawContours(thresh, contours, -1, (127,186,50), 1)

    for contour in contours:
        x,y,w,h = cv2.boundingRect(contour)
        cv2.rectangle(frame2,(x,y),(x+w,y+h),(0,255,0),2)

    # 2) COLOR DETECTION
    hsv = cv2.cvtColor(frame2, cv2.COLOR_BGR2HSV)

    # define range of blue color in HSV
    trshld_low = np.array([-33,100,100])
    trshld_high = np.array([33,255,255])

    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, trshld_low, trshld_high)

    thresh_color = cv2.threshold(mask, 100, 255, cv2.THRESH_BINARY)[1]

    kernel = np.ones((20,20),np.uint8)
    dilated = cv2.dilate(thresh_color,kernel,iterations = 1)
    cv2.imshow('dilated', dilated)


    # Bitwise-AND mask and original image
    masked = cv2.bitwise_and(frame2, frame2, mask = thresh_color)


    
    #edges = cv2.Canny(frame2, 150, 200)

    #final = cv2.addWeighted(thresh, 1, edges, 1, 0)

    cv2.imshow('Frame1', frame1)
    cv2.imshow('Frame2', frame2)
    cv2.imshow('Coloured', masked)
    cv2.imshow('Mask', mask)
    cv2.imshow('thresh_color', thresh_color)
    #cv2.imshow('Final', final)

    out.write(thresh)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()