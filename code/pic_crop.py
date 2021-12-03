from PIL import Image
import face_recognition
import cv2
import numpy as np


# this code need to run on version of python3.6
# use anaconda to install python3.6

for i in range(1,63):
    image = face_recognition.load_image_file(r"C:\Users\wilso\OneDrive\桌面\maskPhoto\6 ("+str(i)+").jpg")

    face_locations = face_recognition.face_locations(image)

    print("{} face(s) in this photograph.".format(len(face_locations)))

    for face_location in face_locations:
         top, right, bottom, left = face_location
         # You can access the actual face itself like this:
         face_image = image[top:bottom, left:right]
         pil_image = Image.fromarray(face_image)
         pil_image.show()
         pil_image = np.array(pil_image)
         cv2.imwrite(r"C:\Users\wilso\OneDrive\桌面\maskPhoto\6 ("+str(i)+").jpg",cv2.cvtColor(pil_image, cv2.COLOR_RGB2BGR))