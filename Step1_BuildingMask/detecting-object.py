
import cv2


# Defining lower bounds and upper bounds of founded Mask

#mask for intel
'''min_blue, min_green, min_red = 67, 64, 84
max_blue, max_green, max_red = 255, 255, 255'''

#mask for nvidia
min_blue, min_green, min_red = 30, 25, 115
max_blue, max_green, max_red = 64, 255, 255

# Getting version of OpenCV that is currently used
# Converting string into the list by dot as separator
# and getting first number
v = cv2.__version__.split('.')[0]

# Defining object for reading video from camera

# Defining loop for catching frames
while True:
    # Capture frame-by-frame from camera
    image_BGR = cv2.imread('objects-to-detect3.jpg')

    # Converting current frame to HSV
    frame_HSV = cv2.cvtColor(image_BGR, cv2.COLOR_BGR2HSV)

    # Implementing Mask with founded colours from Track Bars to HSV Image
    mask = cv2.inRange(frame_HSV,
                       (min_blue, min_green, min_red),
                       (max_blue, max_green, max_red))

    cv2.namedWindow('Binary frame with Mask', cv2.WINDOW_NORMAL)
    cv2.imshow('Binary frame with Mask', mask)


    if v == '3':
        _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    else:
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # Finding the biggest Contour by sorting from biggest to smallest
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # Extracting Coordinates of the biggest Contour if any was found
    if contours:
        # Getting rectangle coordinates and spatial size from biggest Contour
        # Function cv2.boundingRect() is used to get an approximate rectangle
        # around the region of interest in the binary image after Contour was found
        (x_min, y_min, box_width, box_height) = cv2.boundingRect(contours[0])

        # Drawing Bounding Box on the current BGR frame
        cv2.rectangle(image_BGR, (x_min - 15, y_min - 15),
                      (x_min + box_width + 15, y_min + box_height + 15),
                      (0, 255, 0), 3)

        # Preparing text for the Label
        label = 'Detected Object'

        # Putting text with Label on the current BGR frame
        cv2.putText(image_BGR, label, (x_min - 5, y_min - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    # Showing current BGR frame with Detected Object
    # Giving name to the window with Detected Object
    # And specifying that window is resizable
    cv2.namedWindow('Detected Object', cv2.WINDOW_NORMAL)
    cv2.imshow('Detected Object', image_BGR)

    # Breaking the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# Destroying all opened windows
cv2.destroyAllWindows()