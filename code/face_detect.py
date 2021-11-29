import cv2
import numpy as np
from imutils.video import WebcamVideoStream
from os.path import exists
from urllib.request import urlretrieve

# download model if not exists
prototxt = "deploy.prototxt"
caffemodel = "res10_300x300_ssd_iter_140000.caffemodel"
if not exists(prototxt) or not exists(caffemodel):
    urlretrieve(f"https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/{prototxt}", prototxt)
    urlretrieve(f"https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/{caffemodel}", caffemodel)

# initialize model
net = cv2.dnn.readNetFromCaffe(prototxt=prototxt, caffeModel=caffemodel)

# Detect function
# min_confidence: the minimum probability to filter detections, i.e., the minimum probability to keep a detection
def detect(img, min_confidence=0.6):
    # get the height and width of the image
    (h, w) = img.shape[:2]

    # create a 4D blob from the image
    # the parameters are:
    # 1. the input image
    # 2. the scalar value that normalizes the pixel values
    # 3. the width and height of the image
    # 4. b subtract 104, g subtract 117, r subtract 123
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    # setup the input(blob) and get the results
    net.setInput(blob)
    detectors = net.forward()

    # initialize our list of bounding boxes
    bounding_boxs = []

    # loop all the results after detection
    for i in range(0, detectors.shape[2]):
        # get the confidence of the prediction
        confidence = detectors[0, 0, i, 2]

        # filter out weak detections, which is lower than 60%
        if confidence < min_confidence:
            continue

        # box is a list that contains the distances of the left, top, right and bottom of the bounding box
        box = detectors[0, 0, i, 3:7] * np.array([w, h, w, h])
        (x0, y0, x1, y1) = box.astype("int")
        bounding_boxs.append({"box": (x0, y0, x1 - x0, y1 - y0)})

    return bounding_boxs

# main function here
def main():
    # open the webcam
    vs = WebcamVideoStream().start()

    # a start message
    print("system starts")

    while True:
        # get the frame from the webcam
        frame = vs.read()

        # detect the faces, and get the bounding box
        bounding_boxs = detect(frame)

        # loop over the bounding boxes, for each bounding box, draw a rectangle
        for bounding_box in bounding_boxs:
            (x, y, w, h) = bounding_box["box"]

            # results could be different after implementing the mask detection
            # the pseudo code are as follows:
            # if the mask in on:
            #      display "Mask On"
            #      colors of the bounding box and the text will be green
            # else:
            #      display "Mask Off"
            #      colors of the bounding box and the text will be red

            # example: when mask in on
            # draw the bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # display mask on or off message
            text = "Mask On"
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # show the frame
        cv2.imshow("Frame", frame)

        # if the user press 'q', then break
        key = cv2.waitKey(1) & 0xff
        if key == ord('q'):
            break

if __name__ == '__main__':
    main()