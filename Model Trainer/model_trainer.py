"""

This Python script trains the model

"""

from tensorflow.keras import models, layers, callbacks, optimizers, utils
from datetime import datetime
import os, cv2, numpy, random, math
from matplotlib import pyplot
import h5py
from sklearn.model_selection import train_test_split

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

to_categorical = utils.to_categorical

print("start")

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
            
            # .reshape(-1, self.dim, self.dim, 1)
            
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
  
class Model(BasicData):
    
    
    # This function will remove modelExtension from the filename if detected
    
    def __check_name(self, name):
        
        exlen = len(self.modelExtension)
        
        if ((type(name) == str) and (name[ -1 * exlen : ] == self.modelExtension)):
            
            return name[ : -1 * exlen ]
        
        return (str(name) if (type(name) != str) else (name))
    
    def __init__(self, filename):
        
        self.model = models.Sequential()
        #self.model.add(layers.Input(shape = (self.dim, self.dim, 1)))
        self.filename, self.pointer = self.__check_name(filename), 1
        
    def add_layers(self, layerList):
        
        for layer in layerList:
            
            self.model.add(layer)
        
    def compile_model(self, show_summary = False, ** kwargs):
        
        if (show_summary):
            
            self.model.summary()
        
        self.model.compile(**kwargs)        
    
    def fit_model(self, imageList, labelList, ** kwargs):
        
        self.history = self.model.fit(imageList, labelList, ** kwargs)
        
    def save_model(self, meta):
        
        self.model.save(self.filename + meta + self.modelExtension)
        self.pointer += 1
        self.__save_history(meta)
        
    def __save_history(self, meta):
        
        pyplot.plot(self.history.history["loss"])
        
        pyplot.plot(self.history.history["val_loss"])
        
        pyplot.legend(["loss", "val_loss"])
        
        pyplot.savefig(self.filename + meta + "_lossFig.png")
        
        pyplot.show()
        
        pyplot.plot(self.history.history["Accuracy"])
        
        pyplot.plot(self.history.history["val_Accuracy"])
        
        pyplot.legend(["acc", "val_acc"])
        
        pyplot.savefig(self.filename + meta + "_valFig.png")
        
        pyplot.show()
        
maskPath, facePath = "./dataset/withMask/", "./dataset/noMask/"

modelName =  "./../mask_class_model/face_mask_detection.hdf5"

runs, batchScale, shuffle = 1, 1.0, True

batch_size, epochs, validation_split = 128, 100, (1 / 8.0)

layerList = [
    layers.MaxPooling2D(pool_size = (4, 4), padding = "same", input_shape = (160, 160, 1)),
    layers.Conv2D(filters = 32, kernel_size = (3, 3), activation = 'relu'),
    layers.MaxPooling2D(pool_size = (2, 2), padding = "same"),
    layers.Conv2D(filters = 64, kernel_size = (3, 3), activation = "relu"),
    layers.MaxPooling2D(pool_size = (2, 2), padding = "same"),
    layers.Flatten(),
    layers.Dense(128, activation = "relu"),
    layers.Dropout(0.3),
    layers.Dense(64, activation = "relu"),
    layers.Dropout(0.1),
    layers.Dense(32, activation = "relu"),
    layers.Dropout(0.2),
    layers.Dense(1, activation = "sigmoid")
]

model = Model(filename = modelName)

model.add_layers(layerList)

opt = optimizers.Adam(learning_rate = 0.0001)

model.compile_model(show_summary = True, **{"optimizer" : opt, "loss" : "mse", "metrics" : [ "Accuracy" ]})
"""
for run in range(runs):
    
    mask = Images(folderPath = maskPath, batchScale = batchScale, shuffle = shuffle)
    
    face = Images(folderPath = facePath, batchScale = batchScale, shuffle = shuffle)
    
    while ((mask.load_images()) and (face.load_images())):
        
        print(len(mask.imageList), len(face.imageList))
        
        length1 = mask.curBatch
        length2 = face.curBatch

        # split images here
        images = numpy.concatenate((mask.imageList, face.imageList))
        labels = numpy.concatenate((numpy.ones(length1), numpy.zeros(length2)))
        X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size = validation_split, shuffle = shuffle, random_state = 42)

        length1 = int(numpy.sum(y_train))
        length2 = len(y_train) - int(numpy.sum(y_train))

        # 0 for no mask
        # 1 for mask
        class_weight =  {
            0 : (length2 / (length1 + length2)),
            1 : (length1 / (length1 + length2))
        }

        model.fit_model(
            X_train,
            y_train,
            batch_size = batch_size,
            epochs = epochs,
            validation_data = (X_test, y_test),
            shuffle = shuffle,
            callbacks = [ callbacks.EarlyStopping(monitor = "val_Accuracy", patience = 50, verbose = 0) ],
            class_weight = class_weight
        )
        
        model.save_model(f"_{run}_{model.pointer}")
"""