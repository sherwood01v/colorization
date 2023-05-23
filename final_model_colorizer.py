from tensorflow import keras
from keras.layers import Conv2D, UpSampling2D
from keras.models import Sequential
from tensorflow.keras.utils import img_to_array
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
import matplotlib.pyplot as plt

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

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import keras_tuner as kt
from kerastuner.tuners import RandomSearch

def build_model(optimizer, num_layers, filters):
    # Encoder
    inputs = keras.Input(shape=(160, 160, 1))
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', strides=2)(inputs)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', strides=2)(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', strides=2)(x)

    # Intermediate convolutional layers based on the hyperparameter `num_layers`
    for _ in range(num_layers):
        x = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(x)

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

    model.compile(optimizer=optimizer,
                loss='mae',
                metrics=['accuracy'])

    return model

import os
import json

# Folder path containing the trials
folder_path = 'final_model/hyperparamters'

# Iterate over the trials in the folder
best_accuracy = 0
best_trial = None
for foldername in os.listdir(folder_path):
    subfolder_path = os.path.join(folder_path, foldername)
    if os.path.isdir(subfolder_path):
        for filename in os.listdir(subfolder_path):
            if filename.endswith('.json'):
                trial_path = os.path.join(subfolder_path, filename)
                with open(trial_path, 'r') as f:
                    trial_config = json.load(f)
                trial_accuracy = trial_config['score']
                # Compare the trial's loss with the current best loss
                if trial_accuracy > best_accuracy:
                    best_accuracy = trial_accuracy
                    best_trial = subfolder_path

# Get the trial number from the best trial filename
best_trial_number = best_trial.split('/')[2]
print(f"The best hyperparameters are in trial number: {best_trial_number}")

# File name of the trial configuration
trial_file_name = f"{best_trial_number}.json"

# Full path to the trial configuration file
trial_path = os.path.join(f"{folder_path}/{best_trial_number}/trial.json")

# Load the trial configuration
with open(trial_path, 'r') as f:
    trial_config = json.load(f)

# Convert the trial configuration to HyperParameters object
optimizer = trial_config['hyperparameters']['values']['optimizer']
num_layers = trial_config['hyperparameters']['values']['num_layers']
filters = trial_config['hyperparameters']['values']['filters']
batch_size = trial_config['hyperparameters']['values']['batch_size']
shuffle = trial_config['hyperparameters']['values']['shuffle']

model = build_model(optimizer, num_layers, filters)
history = model.fit(X, Y,
                    epochs = 50,
                    batch_size=batch_size,
                    shuffle = shuffle,
                    validation_split=0.2)
val_acc_per_epoch = history.history['val_accuracy']
best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
print('Best epoch: %d' % (best_epoch,))

hypermodel = build_model(optimizer, num_layers, filters)

# Retrain the model
result = hypermodel.fit(X, Y,
            epochs = best_epoch,
            batch_size=batch_size,
            shuffle = shuffle,
            validation_split=0.2)

hypermodel.save("Colorization_model.h5")

colorized_deneme = hypermodel.predict(X[4000:])

eval_result = hypermodel.evaluate(X[4000:], Y[4000:])
print("[test loss, test accuracy]:", eval_result)