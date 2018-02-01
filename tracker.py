import numpy as np
import cv2

cap = cv2.VideoCapture('GOPR1152.MP4')

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
#out = cv2.VideoWriter('output.avi', fourcc, 29.97, (640,240))
frame = []
gray1 = []

green_track = []
red_track = []
plot = np.zeros((720, 1280, 3), np.uint8)

if(cap.isOpened()):
    ret, frame1 = cap.read()

    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

while(cap.isOpened()):

    ret, frame2 = cap.read()

    '''
    # 1) DIFFERENCE DETECTION

    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    diff = cv2.absdiff(gray1, gray2)

    thresh = cv2.threshold(diff, 60, 255, cv2.THRESH_BINARY)[1]

    _, contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print(len(contours))
    contourimg = cv2.drawContours(thresh, contours, -1, (127,186,50), 1)

    for contour in contours:
        x,y,w,h = cv2.boundingRect(contour)
        cv2.rectangle(frame2,(x,y),(x+w,y+h),(0,255,0),2)
    '''

    # 2) COLOR DETECTION


    # a) GREEN CUP
    blurred = cv2.GaussianBlur(frame2, (15, 15),0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # Define range of green color in HSV
    green_low = np.array([20,80,80])
    green_high = np.array([45,130,130])

    # Threshold the HSV image to get only green colors
    green_mask = cv2.inRange(hsv, green_low, green_high)

    # Dilate the points
    kernel = np.ones((5,5),np.uint8)
    green_dilated = cv2.dilate(green_mask, kernel, iterations = 1)
    
    # Find the contours
    _, green_contours, _ = cv2.findContours(green_dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #print("Green contours: " + str(len(green_contours)))
    #contourimg = cv2.drawContours(dilated, contours, -1, (127,186,50), 1)

    #Draw bounding rectangles around the curvevs
    for contour in green_contours:
        x,y,w,h = cv2.boundingRect(contour)
        cv2.rectangle(frame2,(x,y),(x+w,y+h),(0,255,0),2)
        green_track.append((x + w/2, y + h/2))
        plot = cv2.circle(plot, ((int) (x + w/2), (int) (y + h/2)), 3, (0,255,0), 2 )
        




    # b) RED RIBBON
    blurred = cv2.GaussianBlur(blurred, (25, 25),0)
    # Define range of red color in HSV
    red_low = np.array([160,40,40])
    red_high = np.array([180,140,140])

    # Threshold the HSV image to get only red colors
    red_mask = cv2.inRange(hsv, red_low, red_high)

    # Dilate the points
    kernel = np.ones((15,15),np.uint8)
    red_dilated = cv2.dilate(red_mask, kernel, iterations = 1)
    
    # Find the contours
    _, red_contours, _ = cv2.findContours(red_dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #print("Red contours: " + str(len(red_contours)))
    #contourimg = cv2.drawContours(dilated, contours, -1, (127,186,50), 1)

    #Draw bounding rectangles around the curvevs
    for contour in red_contours:
        x,y,w,h = cv2.boundingRect(contour)
        cv2.rectangle(frame2,(x,y),(x+w,y+h),(0,0,255),2)
        red_track.append((x + w/2, y + h/2))
        plot = cv2.circle(plot, ((int) (x + w/2), (int) (y + h/2)), 3, (0,0,255), 2 )



    cv2.imshow('Frame1', frame1)
    cv2.imshow('Frame2', frame2)
    cv2.imshow('Path', plot)

    #out.write(thresh)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print(red_track)
print(green_track)

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()