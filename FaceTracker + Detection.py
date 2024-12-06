from djitellopy import Tello
import time
import cv2
import numpy as np
import cvzone

tello_drone = Tello()
tello_drone.connect()
print(tello_drone.get_battery())

tello_drone.streamon()
tello_drone.takeoff()
tello_drone.send_rc_control(0, 0, 25, 0)
time.sleep(4) # Hard programming the height it flys upwards based on seconds ex: 3.75 seconds UP at speed 25

width, height = 360, 240
for_back_Range = [6500, 7400]
pid = [0.4, 0.4, 0]
pError = 0

detection_threshold = 0.55 # << this is for the confThreshold (confidence threshold) function, which determines whether something is an object
#           or not, right now it is set so that if it is 60% sure or greater,
#           then it will consider something an object >>

nmsThres = 0.2 # the nms threshold will remove any multiples of the same object

classNames = [] # << an empty list we will add all the files (files full of class names) >>
classFile = 'coco.names' # << txt file of 90 different names >>
with open(classFile, 'rt') as f:
    classNames = f.read().split('\n') # << writes all names in the file to the classNames list >>
print(classNames)
# -------------------------------
configPath =  'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = "frozen_inference_graph.pb"
network = cv2.dnn_DetectionModel(weightsPath,configPath) # << loading everything into a network >>
network.setInputSize(320,320)             # /
network.setInputScale(1.0/127.5)          # | Configuring the network settings
network.setInputMean((127.5,127.5,127.5)) # \
network.setInputSwapRB(True) # << opencv library uses BGR (Blue,Green,Red) colorscale instead of RGB,
                                # so this fixes that >>

#THIS CODE IS USED FOR FINDING THE CENTER OF THE FACE (for rotate angle), AND SIZE OF THE AREA OF THE FACE (for distance)
def findFace(img):
    faceCascade = cv2.CascadeClassifier("Resources/haarcascade_frontalface_default.xml")
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # convert the image to grayscale
    faces = faceCascade.detectMultiScale(imgGray, 1.2, 8)

    myFaceListCenter = []
    myFaceListArea = []

    for(x,y,w,h) in faces: # x,y, width, height
        cv2.rectangle(img,(x,y),(x+w,y+h), (0,0,255), 2) # code for the box that surrounds users face
        centerx = x +w // 2
        centery = y + h // 2
        area = w * h
        cv2.circle(img, (centerx,centery), 2, (0, 255, 0), cv2.FILLED) # the Green "crosshair" to indicate the center
        myFaceListCenter.append([centerx, centery])
        myFaceListArea.append(area)
    if len(myFaceListArea) != 0: # basically saying "if there is a face on screen, run the next code"
        i = myFaceListArea.index(max(myFaceListArea))
        return img, [myFaceListCenter[i], myFaceListArea[i]]
    else: #if there is no face, return nothing
        return img, [[0, 0], 0]

# //
#
# Code for the movement of following the face that is recognized on screen
#
# //
def trackFace(info, width, pid, pError):
    area = info[1]
    x,y = info[0]
    forward_backward = 0

    error = x - width // 2
    speed = pid[0] * error + pid[1] * (error-pError)
    speed = int(np.clip(speed, -100, 100))

    # conditions for whether drone needs to move closer, back up, or stay still
    if area > for_back_Range[0] and area < for_back_Range[1]:
        forward_backward = 0 #Drone does not need to move forward or backward
    elif area > for_back_Range[1]:
        forward_backward = -15 #Drone is too far away
    elif area < for_back_Range[0] and area != 0:
        forward_backward = int((((for_back_Range[0]-area) / 5200) * 50) -20)

    if x == 0:
        speed = 0
        error = 0

    print(speed, forward_backward)

    tello_drone.send_rc_control(0, forward_backward, 0, speed)
    return error

# Accessing the camera, creating a window of the real time recording
# window = cv2.VideoCapture(0)
while True:
   # _, img = window.read()
    img = tello_drone.get_frame_read().frame #Connect to the drone camera in real time
    img = cv2.resize(img, (width, height))
    img, info = findFace(img)  # Creates a red box around the detected faces and places a green dot in the center of the face.
    pError = trackFace(info, width, pid, pError)
    classIds, confidence_values, bbox = network.detect(img, confThreshold=detection_threshold, nmsThreshold=nmsThres)  # see named constants

    try:
        for classIds, confidence_values, box in zip(classIds.flatten(), confidence_values.flatten(), bbox):
        # print(classIds,confidence_values,box) # this is a list of all the objects available
            cvzone.cornerRect(img, box, rt=0)
            cv2.putText(img, f'{classNames[classIds - 1].upper()} {round(confidence_values * 100, 2)}',
                        (box[0] + 10, box[1] + 30), cv2.FONT_HERSHEY_TRIPLEX,
                        1, (0, 255, 0), 2)
    except:
        pass  # <<_This code is lazy AF, basically if it detects nothing it panics and gives back an error but the try
    # and except statement fixes that >>
    # print("Area of Face(In Pixels)", info[1])
    # print("Center of Face(x and y)", info[0])

    cv2.imshow("Camera Output", img)
    if cv2.waitKey(1) & 0xFF == ord('q'): # the drone will land if you hit q
        tello_drone.land()
        break
