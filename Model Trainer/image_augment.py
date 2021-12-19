"""

This Python script is used to generate the augmented images.

"""

from tensorflow.keras import preprocessing, utils
import os, cv2, numpy, sys
img_to_array = utils.img_to_array
ImageDataGenerator = preprocessing.image.ImageDataGenerator
maskPath = "./dataset/withMask/"

filename = [ maskPath + name for name in os.listdir(maskPath) ]

generator = ImageDataGenerator(
    horizontal_flip = True
)

def generate(image):
    sample = numpy.expand_dims(img_to_array(image), 0)
    it = generator.flow(sample, batch_size = 1)
    imageList = []
    for i in range(1):
        batch = it.next()
        image = batch[0].astype("uint8")
        imageList.append(image)
    return imageList

for name in filename:
    image = cv2.imread(name)
    imageList = cv2.flip(image, 1)
    cv2.imwrite(name + ".jpg", imageList)