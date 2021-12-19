import cv2
import numpy as np
from imutils.video import WebcamVideoStream
from os.path import exists
from urllib.request import urlretrieve
from tensorflow.keras import models
import h5py

# change your path here
absolutePath = "C:/Path/Here"

# download model if not exists
prototxt = absolutePath + "face-mask-detection/deploy.prototxt"
caffemodel = absolutePath + "face-mask-detection/res10_300x300_ssd_iter_140000.caffemodel"

# initialize model
net = cv2.dnn.readNetFromCaffe(prototxt=prototxt, caffeModel=caffemodel)

# Detect function
# min_confidence: the minimum probability to filter detections, i.e., the minimum probability to keep a detection
def detect(img, min_confidence = 0.6):
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

modelPath = "./Model/face_mask_detection.hdf5"

model = models.load_model(modelPath)

# main function here
def main():
    # a preparing message
    print("system preparing...")

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
            
            # get the centry point of the bounding box
            center_point = (x + w / 2, y + h / 2)
            # scale the bounding box by 0.8
            scale = 0.85
            # resize the bounding box
            input_w = int(w * scale)
            input_h = int(h * scale)
            input_x = int(center_point[0] - input_w / 2)
            input_y = int(center_point[1] - input_h / 2)
            # crop the bounding box
            crop_img = frame[input_y : input_y + input_h , input_x : input_x + input_w]
            # resize the crop_img to 160 * 160
            crop_img = cv2.resize(crop_img, (160, 160), interpolation=cv2.INTER_AREA)
            crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY) / 255.0
                        
            # generate result
            # take crop_img as the input of the model
            mask_confident = np.array(model(np.stack([crop_img])))[0][0]
            
            # display mask on or off message
            if(round(mask_confident * 100, 2) >= 50):
                text = "Mask On"
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, str(text + " ") + (str(round(mask_confident * 100, 2)) + "%"), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                text = "No Mask"
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(frame, str(text + " ") + (str(round(100 - mask_confident * 100, 2)) + "%"), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    
        # show the frame
        cv2.imshow("Frame", frame)
        
        # if the user press 'q', then break
        key = cv2.waitKey(1) & 0xff
        if key == ord('q'):
            break
    
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()