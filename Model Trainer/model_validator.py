"""

This Python Script gets the results after predicting the images(dataset).

"""

from tensorflow.keras import models, layers, callbacks
from datetime import datetime
import os, cv2, numpy, random, math, pandas, time
from matplotlib import pyplot
np = numpy
random.seed(datetime.now())

class BasicData:
    
    
    """
    
        The dimension of jpeg files: 
            
            (1080, 1080) when dim = 1080
            ( 300,  300) when dim =  300
            ( 160,  160) when dim =  160
        
    """
    
    dim = 160
    
    
    """
    
        The extension of neural network models.
        
    """
    
    modelExtension = ".hdf5"

class Images(BasicData):
    
    
    # This function confines batchScale to ( 0.0, 1.0 ]
    
    def __check_scale(self, scale):
        
        
        """
            
            To ensure that at least 1 file will be loaded when there are some files available,
        we pick an arbitrarily-small number 1e-10 as the lower bound of batchScale instead of 0.
        
            => [ 1e-10, 1.0 ] ≈ ( 0.0, 1.0 ] 
        
        """
        
        return max(1e-10, min(scale, 1.0))
    
    
    # This function identifies whether or not a file name is that of a jpeg one
    
    def __is_jpeg(self, name):
        
        return ((name[-4:].lower() == ".jpg") or (name[-5:].lower() == ".jpeg"))
    
    def __init__(self, folderPath = "./", batchScale = 1.0, shuffle = False):
        
        
        # The names of all jpeg files inside the folder are found and placed in a list
        
        self.nameList = [ 
            folderPath + name for name in os.listdir(folderPath) \
                if (self.__is_jpeg(name) and os.path.isfile(folderPath + name))
        ]
        
            
        # Shuffling the list containing the names of jpeg files
        
        if (shuffle):
            random.shuffle(self.nameList)
        
        
        # The number of jpeg files found in the folder
        #   An index pointing to the beginning of the next batch
        
        self.total, self.pointer = len(self.nameList), 0
        
        
        # The maximum number of jpeg files to be loaded at a time
        
        self.batch = math.ceil(self.total * self.__check_scale(batchScale))
        
    def load_images(self):
        
        self.imageList = []
        
        
        # No images will be loaded when the pointer has reached the end of the list
        
        if (self.pointer < self.total):
            
            
            # Loading jpeg files in grayscale mode and normalizing pixel data from [ 0, 255 ] to [ 0.0, 1.0 ]
            
            self.imageList = [
                cv2.imread(name, cv2.IMREAD_GRAYSCALE) / 255.0 \
                    for name in self.nameList[ self.pointer : self.pointer + self.batch ]
            ]
                
            
            # Converting [ numpy.array([1]), numpy.array([2]) ] to numpy.array([ [1], [2] ])
            
            self.imageList = numpy.stack(self.imageList, axis = 0)
                
            
            # Updating the pointer and bounding it within [ 0, self.total ]
            
            self.pointer = min(self.pointer + self.batch, self.total)
        
        
        # Saving the number of images in the current batch
        
        self.curBatch = len(self.imageList)
        
        return self.curBatch
        
maskPath, facePath, modelName = "./dataset/withMask/", "./dataset/noMask/", "./../mask_class_model/test_model_E_0_1.hdf5"

testPath = "C:/Users/ailab/Desktop/project/ML-maskDetectionProject/face-mask-detection/code/"


batchScale, shuffle = 1.0, False

model = models.load_model("C:/Users/ailab/Desktop/project/ML-maskDetectionProject/face-mask-detection/code/mask_class_model/test_model_E_0_1.hdf5")

test = Images(folderPath=testPath, batchScale=batchScale, shuffle=shuffle)

test.load_images()

# print(numpy.array(model(  test.imageList )))

mask = Images(folderPath = maskPath, batchScale = batchScale, shuffle = shuffle)

face = Images(folderPath = facePath, batchScale = batchScale, shuffle = shuffle)

mask.load_images(), face.load_images()

pred_mask = pandas.DataFrame(numpy.array(model.predict(mask.imageList)), columns = ["result"])

pred_face = pandas.DataFrame(numpy.array(model.predict(face.imageList)), columns = ["result"])

pred_mask.to_csv("./../mask_class_model/pred_mask.csv", index = False)

pred_face.to_csv("./../mask_class_model/pred_face.csv", index = False)

test01 = model.predict(mask.imageList)
test01 = numpy.array(test01)
for i in range(len(test01)):
    if test01[i][0] > 0.5:
        test01[i][0] = 1
    else:
        test01[i][0] = 0
print(np.sum(test01) / len(test01))

test02 = model.predict(face.imageList)
test02 = numpy.array(test02)
for i in range(len(test02)):
    if test02[i][0] > 0.5:
        test02[i][0] = 1
    else:
        test02[i][0] = 0
print((len(test02) - np.sum(test02)) / len(test02))