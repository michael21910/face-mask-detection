"""

This Python script is used to convert images to gray scale.

"""

import cv2

# change the number here if the dataset has changed
number_of_images = 6000

# convert BGR to Grayscale
def convert(img):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return imgGray

def readImageAndSave160(n):
    for i in range(1, n + 1):
        img = cv2.imread('Path/Here/1_(%d).jpg'%(i))
        img = convert(img)
        cv2.imwrite('Path/Here/1_(%d).jpg'%(i), img)
    for i in range(1,n+1):
        img = cv2.imread('Path/Here/0_(%d).jpg'%(i))
        img = convert(img)
        cv2.imwrite('Path/Here/0_(%d).jpg'%(i), img)

def get_data():
    readImageAndSave160(number_of_images)

get_data()