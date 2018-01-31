import numpy as np
import cv2

cap = cv2.VideoCapture('DSC_0354.MOV')

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 29.97, (640,240))

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        # Convert BGR to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # define range of blue color in HSV
        trshld_low = np.array([-33,100,100])
        trshld_high = np.array([33,255,255])

        # Threshold the HSV image to get only blue colors
        mask = cv2.inRange(hsv, trshld_low, trshld_high)

        # Bitwise-AND mask and original image
        masked = cv2.bitwise_and(frame, frame, mask = mask)

        #blrd = cv2.GaussianBlur(frame, (3, 3), 0)

        edges = cv2.Canny(frame, 100, 200)


        _, contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        cv2.imshow('Frame', frame)

        contourimg = cv2.drawContours(frame, contours, -1, (127,186,50), 1)

        cv2.imshow('Mask', mask)
        cv2.imshow('Masked', masked)
        cv2.imshow('Edges', edges)
        cv2.imshow('Cotours', contourimg)

        out.write(masked)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()