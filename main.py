import cv2
import numpy as np

# reading the video using video capture object
vehicle_vid = cv2.VideoCapture("C:\\Users\Rahul Jain\\Desktop\\Opencv project\\6 vehicle detector\\final.mp4")

# cascade function  to detect cars in frames
car_cascade = cv2.CascadeClassifier('C:\\Users\\Rahul Jain\\Downloads\\Car_Detection_System-master\\Car_Detection_System-master\\cars.xml')

# while loop to display frames continuously so that it will create video
while True:
    # check is a boolean data type, it returns True if python is able to read the VideoCapture()
    # frame is a numpy array, it represents one frame at a time which is captured
    check, frame = vehicle_vid.read()
    # prints frames in the form of numpy array
    print(frame)
    # converts one frame at a time to a grayscale image
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cars = car_cascade.detectMultiScale(gray,1.05,5)

    # this function is used to create a rectangular box around the detected cars, co-ordinates, color of rectangle are mentioned
    for x,y,w,h in cars:
        cv2.rectangle(frame, (x,y),(x+w,y+h),(0,0,255),2)

        # function to display frames of video
        cv2.imshow("Vehichle detection", frame)


    # in case of image we use waitkey function to display an image for specific time
    # in case of video image means frame, and since it is in loop after 1 frame goes 2nd frame comes continuously.
    key = cv2.waitKey(1)


    # if statement to close the window using key x
    if key == ord('x'):
        break


vehicle_vid.release()
