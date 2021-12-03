from PIL import Image
import cv2

basewidth = 400
for i in range (1,2793):
    path = r"C:\Users\wilso\OneDrive\桌面\dataadjust\dataset400 - 複製\noMask\noMask\0_("+str(i)+").jpg"
    img = Image.open(path)
    #wpercent = (basewidth / float(img.size[0]))
    #hsize = int((float(img.size[1]) * float(wpercent)))
    img = img.resize((basewidth, basewidth), Image.ANTIALIAS)
    print(i)
    img = img.convert("RGB")
    img.save(r"C:\Users\wilso\OneDrive\桌面\dataadjust\dataset400 - 複製\noMask\noMask\0_("+str(i)+").jpg")