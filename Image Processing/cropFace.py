"""

This Python script is used to crop the face from the image.

"""

from PIL import Image
import face_recognition
import cv2
import numpy as np

# only avaliable on python 3.6

# change the number here if the dataset has changed
number_of_images = 6000
for i in range(1, number_of_images + 1):
    image = face_recognition.load_image_file(r"C:/Path/Here" + str(i) + ").jpg")

    face_locations = face_recognition.face_locations(image)

    print("{} face(s) in this photograph.".format(len(face_locations)))

    for face_location in face_locations:
         top, right, bottom, left = face_location
         face_image = image[top:bottom, left:right]
         pil_image = Image.fromarray(face_image)
         pil_image.show()
         pil_image = np.array(pil_image)
         pil_image = cv2.resize(pil_image, (160, 160))
         cv2.imwrite(r"C:/Path/Here" + str(i) + ").jpg", cv2.cvtColor(pil_image, cv2.COLOR_RGB2GRAY))