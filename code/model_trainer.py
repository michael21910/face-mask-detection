from tensorflow.keras import models, layers, callbacks
from datetime import datetime
import os, cv2, numpy, random, math
#
from matplotlib import pyplot
import time

random.seed(datetime.now())

class BasicData:
    
    # Dimension of jpeg files: (300, 300) when dim = 300
    dim = 160
    
    modelExtension = ".h5"

class Images(BasicData):
    
    # Dimension of jpeg files: (300, 300) when dim = 300
    #dim = 300
    
    # Making sure batchScale is within [ 1e-10, 1.0 ]
    # 1e-10 is an arbitrarily-small number which can be rounded up to 1 using math.ceil()
    def __check_scale(self, scale):
        
        return max(1e-10, min(scale, 1.0))
    
    # This function is used to filter out all but jpeg files
    def __is_jpeg(self, name):
        
        return ((name[-4:].lower() == ".jpg") or (name[-5:].lower() == ".jpeg"))
    
    def __init__(self, folderPath = "./", batchScale = 1.0, shuffle = False):
        
        # A list of all jpeg files within the target folder will be created
        self.nameList = [ 
            folderPath + name for name in os.listdir(folderPath) \
                if (self.__is_jpeg(name) and os.path.isfile(folderPath + name))
        ]
        
        # Shuffling the list of jpeg file names
        if (shuffle):
            random.shuffle(self.nameList)
        
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
  
class Model(BasicData):
    
    def __check_name(self, name):
        
        exlen = len(self.modelExtension)
        
        if ((type(name) == str) and (name[ -1 * exlen : ] == self.modelExtension)):
            
            return name[ : -1 * exlen ]
        
        return (str(name) if (type(name) != str) else (name))
    
    def __init__(self, filename):
        
        self.model = models.Sequential()
        self.model.add(layers.Input(shape = (self.dim, self.dim, 1)))
        self.filename, self.pointer = self.__check_name(filename), 1
        
    def add_layers(self, layerList):
        
        for layer in layerList:
            
            self.model.add(layer)
        
    def compile_model(self, show_summary = False, ** kwargs):
        
        if (show_summary):
            
            self.model.summary()
        
        self.model.compile(**kwargs)        
    
    def fit_model(self, imageList, labelList, ** kwargs):
        
        self.model.fit(imageList, labelList, ** kwargs)
        
    def save_model(self, meta):
        
        self.model.save(self.filename + meta + self.modelExtension)
        self.pointer += 1
        
maskPath, facePath, modelName = "./mask/", "./face/", "./model/test_model.h5"

runs, batchScale, shuffle = 10, 0.3, True

batch_size, epochs, validation_split = 64, 10, 0.25

mask = Images(folderPath = maskPath, batchScale = batchScale, shuffle = shuffle)

face = Images(folderPath = facePath, batchScale = batchScale, shuffle = shuffle)

layerList = [
    layers.Conv2D(filters = 16, kernel_size = (7, 7), activation = "relu"),
    layers.MaxPooling2D(pool_size = (2, 2), padding = "same"),
    layers.Conv2D(filters = 32, kernel_size = (5, 5), activation = "relu"),
    layers.MaxPooling2D(pool_size = (2, 2), padding = "same"),
    layers.Conv2D(filters = 64, kernel_size = (5, 5), activation = "relu"),
    layers.MaxPooling2D(pool_size = (2, 2), padding = "same"),
    layers.Conv2D(filters = 128, kernel_size = (3, 3), activation = "relu"),
    layers.MaxPooling2D(pool_size = (2, 2), padding = "same"),
    layers.Conv2D(filters = 256, kernel_size = (3, 3), activation = "relu"),
    layers.MaxPooling2D(pool_size = (2, 2), padding = "same"),
    layers.Flatten(),
    layers.Dense(512, activation = "relu"),
    layers.Dropout(0.2),
    layers.Dense(32, activation = "relu"),
    layers.Dropout(0.2),
    layers.Dense(1, activation = "sigmoid")
]

model = Model(filename = modelName)

model.add_layers(layerList)

model.compile_model(show_summary = True, **{"optimizer" : "Adam", "loss" : "mse", "metrics" : [ "Accuracy" ]})

time.sleep(20)

for run in range(runs):
    
    while ((mask.load_images()) and (face.load_images())):
        
        length = mask.curBatch
        
        model.fit_model(
            numpy.concatenate((mask.imageList, face.imageList)),
            numpy.concatenate((numpy.ones(shape = (length)), numpy.zeros(shape = (length)))),
            batch_size = batch_size,
            epochs = epochs,
            validation_split = validation_split,
            shuffle = shuffle,
            callbacks = [ callbacks.EarlyStopping(monitor = "val_loss", patience = 3, verbose = 0) ]
        )
        
        model.save_model(f"_{run}_{model.pointer}")

"""

# Testing and Debugging

maskPath, facePath = "./", "./"

mask, face = Images(folderPath = maskPath, batchScale = 0.3), Images(folderPath = facePath, batchScale = 0.3)

print(f"{mask.total}, {face.total}")
    
while (mask.load_images()):
    for index in range(len(mask.imageList)):
        pyplot.imshow(mask.imageList[index], cmap = "gray")
        pyplot.plot()
    print(len(mask.imageList))
    pyplot.show()
    time.sleep(2)
    
"""