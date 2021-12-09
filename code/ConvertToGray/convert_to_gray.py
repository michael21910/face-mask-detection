import cv2



# convert the BGR to Grayscale
def convert(img):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return imgGray




def readImageAndSave32(n):
    for i in range(1,n+1):
        img = cv2.imread('Dataset\\32x32\withMask\\1_(%d).jpg'%(i))
        img = convert(img)
        cv2.imwrite('Dataset\\32x32\one\\1_(%d).jpg'%(i), img)
    for i in range(1,n+1):
        img = cv2.imread('Dataset\\32x32\\noMask\\0_(%d).jpg'%(i))
        img = convert(img)
        cv2.imwrite('Dataset\\32x32\\zero\\0_(%d).jpg'%(i), img)

def readImageAndSave64(n):
    for i in range(1,n+1):
        img = cv2.imread('Dataset\\64x64\withMask\\1_(%d).jpg'%(i))
        img = convert(img)
        cv2.imwrite('Dataset\\64x64\one\\1_(%d).jpg'%(i), img)
    for i in range(1,n+1):
        img = cv2.imread('Dataset\\64x64\\noMask\\0_(%d).jpg'%(i))
        img = convert(img)
        cv2.imwrite('Dataset\\64x64\\zero\\0_(%d).jpg'%(i), img)

def readImageAndSave80(n):
    for i in range(1, n + 1):
        img = cv2.imread('Dataset\80x80\withMask\\1_ (%d).jpg'%(i))
        img = convert(img)
        cv2.imwrite('Dataset\\80x80\one\\1_(%d).jpg'%(i), img)
    for i in range(1, n + 1):
        img = cv2.imread('Dataset\80x80\\noMask\\0_(%d).jpg'%(i))
        img = convert(img)
        cv2.imwrite('Dataset\\80x80\\zero\\0_(%d).jpg'%(i), img)

def readImageAndSave160(n):
    for i in range(1,n+1):
        img = cv2.imread('Dataset\\160x160\withMask\\1_(%d).jpg'%(i))
        img = convert(img)
        cv2.imwrite('Dataset\\160x160\one\\1_(%d).jpg'%(i), img)
    for i in range(1,n+1):
        img = cv2.imread('Dataset\\160x160\\noMask\\0_(%d).jpg'%(i))
        img = convert(img)
        cv2.imwrite('Dataset\\160x160\\zero\\0_(%d).jpg'%(i), img)




def get_data():
    readImageAndSave32(2792)
    readImageAndSave64(2792)
    readImageAndSave80(2792)
    readImageAndSave160(2792)



get_data()




