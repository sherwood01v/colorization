#import all necessary libraries that gonna be used in the code
#this code was run in conda environment with installed gpu/tensorflow libraries inside
#Graphics Processor: NVIDIA GeForce RTX 3060
#GPU UUID: GPU-2dd7c8db-2b99-c85b-8fe5-d33b14e19629
from tensorflow import keras
from keras.layers import Conv2D, UpSampling2D
from keras.models import Sequential
from tensorflow.keras.utils import img_to_array, load_img
from keras.preprocessing.image import ImageDataGenerator
from skimage.color import rgb2lab, lab2rgb
from skimage.transform import resize
from skimage.io import imsave
import numpy as np
import tensorflow as tf
import os
import re
from tqdm import tqdm
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import keras_tuner as kt
from kerastuner.tuners import RandomSearch

# to get the files in proper order
def sorted_alphanumeric(data):  
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)',key)]
    return sorted(data,key = alphanum_key)
# defining the size of the image
SIZE = 160
color_img = []
path = '/home/***/kchb/color'
files = os.listdir(path)
files = sorted_alphanumeric(files)
for i in tqdm(files):    
    if i == '6000.jpg':
        break
    else:    
        src = cv2.imread(path + '/'+i,1)
        # open cv reads images in BGR format so we have to convert it to RGB
        img = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
        #resizing image
        img = cv2.resize(img, (SIZE, SIZE))
        img = img.astype('float32') / 255.0
        color_img.append(img_to_array(img))

X =[]
Y =[]
for img in color_img:
  try:
      lab = rgb2lab(img)
      X.append(lab[:,:,0]) 
      Y.append(lab[:,:,1:] / 128) #A and B values range from -127 to 128, 
      #so we divide the values by 128 to restrict values to between -1 and 1.
  except:
     print('error')
X = np.array(X)
Y = np.array(Y)
X = X.reshape(X.shape+(1,)) #dimensions to be the same for X and Y
print(X.shape)
print(Y.shape)

class MyHyperModel(kt.HyperModel):
    def build(self, hp):
        # Encoder
        inputs = keras.Input(shape=(160, 160, 1))
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', strides=2)(inputs)
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', strides=2)(x)
        x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', strides=2)(x)

        # Intermediate convolutional layers based on the hyperparameter `num_layers`
        num_layers = hp.Int('num_layers', min_value=1, max_value=5, step=1)
        for _ in range(num_layers):
            x = layers.Conv2D(hp.Choice('filters', values=[256, 512]), (3, 3), activation='relu', padding='same')(x)

        # Decoder
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.UpSampling2D((2, 2))(x)
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.UpSampling2D((2, 2))(x)
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
        x = layers.Conv2D(2, (3, 3), activation='tanh', padding='same')(x)
        outputs = layers.UpSampling2D((2, 2))(x)

        model = keras.Model(inputs, outputs)

        model.compile(optimizer=hp.Choice('optimizer', ['adam', 'rmsprop']),
                    loss='mae',
                    metrics=['accuracy'])

        return model
    
    def fit(self, hp, model, *args, **kwargs):
        return model.fit(
            *args,
            batch_size=hp.Choice("batch_size", [8, 16, 32]),
            shuffle=hp.Boolean("shuffle"),
            **kwargs,
        )

#Random search hyperparameter tuning
tuner = RandomSearch(
    MyHyperModel(),
    objective="val_accuracy",
    max_trials=5,
    #overwrite=True,
    directory="layers",
    project_name="conv_layers",
)

#Earl stopping algorithm (wont proceed unlesss)
stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

#Hyperparameter search
tuner.search(X, Y, epochs=5, validation_split=0.2, callbacks=[stop_early])

# Get the best hyperparameters
best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]

# Print the actual hyperparameters
print("Best Hyperparameters:")
print(f"Batch size: {best_hyperparameters.get('batch_size')}")
print(f"Optimizer: {best_hyperparameters.get('optimizer')}")
print(f"Shuffle: {best_hyperparameters.get('shuffle')}")
print(f"Shuffle: {best_hyperparameters.get('num_layers')}")

model = tuner.hypermodel.build(best_hyperparameters)
#model.fit(X,Y,validation_split=0.1, epochs=5, batch_size=16, verbose = 1)
history = model.fit(X,Y, epochs=5, validation_split=0.2)
val_acc_per_epoch = history.history['val_accuracy']
best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
print('Best epoch: %d' % (best_epoch,))

hypermodel = tuner.hypermodel.build(best_hyperparameters)

# Retrain the model
hypermodel.fit(X, Y, epochs=best_epoch, validation_split=0.2)

eval_result = hypermodel.evaluate(X[4000:])
print("[test loss, test accuracy]:", eval_result)