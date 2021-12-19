"""

This Python script is used to resize the image to 160 x 160(pixels).

"""

from PIL import Image

# change the number here if the dataset has changed
number_of_images = 6000
basewidth = 160
for i in range (1, number_of_images + 1):
    # change the path here
    path = r"C:/Path/Here/0_(" + str(i) + ").jpg"
    img = Image.open(path)
    img = img.resize((basewidth, basewidth), Image.ANTIALIAS)
    img = img.convert("RGB")
    # change the path here
    img.save(r"C:/Path/Here/0_("+str(i)+").jpg")
    # print(i)