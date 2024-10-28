import cv2
import numpy as np
import imutils

# Function to do nothing, used for trackbars
def nothing(x):
    pass

# Create a window
cv2.namedWindow('HSV Trackbar')

# Create trackbars for HSV values
cv2.createTrackbar('H Lower', 'HSV Trackbar', 0, 179, nothing)
cv2.createTrackbar('H Upper', 'HSV Trackbar', 179, 179, nothing)
cv2.createTrackbar('S Lower', 'HSV Trackbar', 0, 255, nothing)
cv2.createTrackbar('S Upper', 'HSV Trackbar', 255, 255, nothing)
cv2.createTrackbar('V Lower', 'HSV Trackbar', 0, 255, nothing)
cv2.createTrackbar('V Upper', 'HSV Trackbar', 255, 255, nothing)

# Capture video from the camera
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the camera
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to HSV
    frame = cv2.flip(frame,1)
    frame = imutils.resize(frame, width=600)
    frame = cv2.GaussianBlur(frame,(71,71),0)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Get the current positions of the trackbars
    h_lower = cv2.getTrackbarPos('H Lower', 'HSV Trackbar')
    h_upper = cv2.getTrackbarPos('H Upper', 'HSV Trackbar')
    s_lower = cv2.getTrackbarPos('S Lower', 'HSV Trackbar')
    s_upper = cv2.getTrackbarPos('S Upper', 'HSV Trackbar')
    v_lower = cv2.getTrackbarPos('V Lower', 'HSV Trackbar')
    v_upper = cv2.getTrackbarPos('V Upper', 'HSV Trackbar')

    # Create a mask based on the current HSV values
    lower_bound = np.array([h_lower, s_lower, v_lower])
    upper_bound = np.array([h_upper, s_upper, v_upper])
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    mask = cv2.erode(mask,None,iterations = 4)
    mask = cv2.dilate(mask,None,iterations = 4)

        #blurlamak noiseyi ve detaili dusurur
        #The kernel size must be positive and odd (e.g., 3, 5, 7, 9, 11).
        #daha buyuk deger daha cok blur demek 
        #biz blurred imageyi islicez ama asil frameyi display edecegiz


    # Bitwise-AND mask and original image to extract the ball
    result = cv2.bitwise_and(frame, frame, mask=mask)

    # Display the results
    cv2.imshow('Original Frame', frame)
    cv2.imshow('Mask', mask)
    cv2.imshow('Detected Ball', result)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()