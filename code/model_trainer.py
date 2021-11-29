from tensorflow.keras import models, layers
from matplotlib import pyplot
import os, cv2, numpy, time, math
class Images:
    
    # Dimension of jpeg files: (300, 300) when dim = 300
    dim = 300
    
    # Making sure batchScale is within [ 1e-10, 1.0 ]
    # 1e-10 is an arbitrarily-small number which can be rounded up to 1 using math.ceil()
    def __check_scale(self, scale):
        
        return max(1e-10, min(scale, 1.0))
    
    # This function is used to filter out all but jpeg files
    def __is_jpeg(self, name):
        
        return ((name[-4:].lower() == ".jpg") or (name[-5:].lower() == ".jpeg"))
    
    def __init__(self, folderPath = "./", batchScale = 1.0):
        
        # A list of all jpeg files within the target folder will be created
        self.nameList = [ 
            folderPath + name for name in os.listdir(folderPath) \
                if (self.__is_jpeg(name) and os.path.isfile(folderPath + name))
        ]
        
        # The total number of jpeg files in the folder 
        #   The index where we should start next
        self.total, self.pointer = len(self.nameList), 0
        
        # The number of jpeg files to be loaded at a time
        self.batch = math.ceil(self.total * self.__check_scale(batchScale))
        
    def load_images(self):
        
        self.imageList = []
        
        # Stop loading jpeg files when the pointer has reached the end
        if (self.pointer < self.total):
            
            # Loading jpeg file in grayscale and normalizing pixels from [0, 255] to [0, 1]
            self.imageList = [
                cv2.imread(name, cv2.IMREAD_GRAYSCALE) / 255.0 \
                    for name in self.nameList[ self.pointer : self.pointer + self.batch ]
            ]
                
            # Converting a list of numpy arrays | (dim, dim) x n | to a numpy array | (n, dim, dim) |
            self.imageList = numpy.stack(self.imageList, axis = 0)
                
            # Updating the pointer
            self.pointer = min(self.pointer + self.batch, self.total)
        
        # Saving the number of jpeg files within the current batch
        self.curBatch = len(self.imageList)
        
        return self.curBatch
    
maskPath, facePath = "./", "./"

mask, face = Images(folderPath = maskPath, batchScale = 0.3), Images(folderPath = facePath, batchScale = 0.3)

"""

# Testing and Debugging

print(f"{mask.total}, {face.total}")
    
while (mask.load_images()):
    for index in range(len(mask.imageList)):
        pyplot.imshow(mask.imageList[index], cmap = "gray")
        pyplot.plot()
    print(len(mask.imageList))
    pyplot.show()
    time.sleep(2)
    
"""