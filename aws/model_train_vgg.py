#coding=utf8
from keras import backend as K
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout, Activation
from keras.optimizers import rmsprop, SGD
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
#from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D 
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Input, Dense
from keras import regularizers
import os
import numpy as np
import pandas as pd
import numpy as np
import pickle
import json

# Set values

num_classes = 9
image_size = 256
nb_epoch = 20
batch_size = 32

train_data_dir = 'data/train'
validation_data_dir = 'data/validation'

nb_train_samples = 120000
nb_validation_samples = 42000

if K.image_data_format() == 'channels_first':
    input_shape = (3, image_size, image_size)
else:
    input_shape = (image_size, image_size, 3)

# Specify model


# instantiate Sequential model
model = Sequential()  
  
model.add(ZeroPadding2D((1,1),input_shape=input_shape))  
model.add(Convolution2D(64, 3, 3, activation='relu'))  
model.add(ZeroPadding2D((1,1)))  
model.add(Convolution2D(64, 3, 3, activation='relu'))  
model.add(MaxPooling2D((2,2), strides=(2,2)))  


model.add(ZeroPadding2D((1,1)))  
model.add(Convolution2D(128, 3, 3, activation='relu'))  
model.add(ZeroPadding2D((1,1)))  
model.add(Convolution2D(128, 3, 3, activation='relu'))  
model.add(MaxPooling2D((2,2), strides=(2,2)))  


model.add(ZeroPadding2D((1,1)))  
model.add(Convolution2D(256, 3, 3, activation='relu'))  
model.add(ZeroPadding2D((1,1)))  
model.add(Convolution2D(256, 3, 3, activation='relu'))  
model.add(ZeroPadding2D((1,1)))  
model.add(Convolution2D(256, 3, 3, activation='relu'))  
model.add(MaxPooling2D((2,2), strides=(2,2)))  

model.add(ZeroPadding2D((1,1)))  
model.add(Convolution2D(512, 3, 3, activation='relu'))  
model.add(ZeroPadding2D((1,1)))  
model.add(Convolution2D(512, 3, 3, activation='relu'))  
model.add(ZeroPadding2D((1,1)))  
model.add(Convolution2D(512, 3, 3, activation='relu'))  
model.add(MaxPooling2D((2,2), strides=(2,2)))  


model.add(ZeroPadding2D((1,1)))  
model.add(Convolution2D(512, 3, 3, activation='relu'))  
model.add(ZeroPadding2D((1,1)))  
model.add(Convolution2D(512, 3, 3, activation='relu'))  
model.add(ZeroPadding2D((1,1)))  
model.add(Convolution2D(512, 3, 3, activation='relu'))  
model.add(MaxPooling2D((2,2), strides=(2,2)))  


model.add(Flatten())  
model.add(Dense(4096, activation='relu'))  
model.add(Dropout(0.5))  
model.add(Dense(4096, activation='relu'))  
model.add(Dropout(0.5))  
#model.add(Dense(1000, activation='softmax'))  

model.add(Dense(num_classes, activation='softmax'))
#compile model

model.summary()

#事实证明，对于分类问题，使用交叉熵(cross entropy)作为损失函数更好些
model.compile(
    loss='categorical_crossentropy',
    optimizer=SGD(lr=0.1),
    metrics=['accuracy']
)


# Image generators
train_datagen = ImageDataGenerator(rescale= 1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(image_size, image_size),
    shuffle=True,
    batch_size=batch_size,
    class_mode='categorical'
    )

validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(image_size, image_size),
    batch_size=batch_size,
    shuffle=True,
    class_mode='categorical'
    )

def generate_arrays_from_file(generator):
    for x,y in generator:
        # x = x.reshape(batch_size, image_size*image_size*3)
        x = x.reshape(len(x), image_size*image_size*3)
        yield (x,y)



# Fit model
# history = model.fit_generator(generate_arrays_from_file(train_generator),
history = model.fit_generator(train_generator,
                    steps_per_epoch=(nb_train_samples // batch_size),
                    epochs=nb_epoch,
                    #validation_data=generate_arrays_from_file(validation_generator),
                    validation_data=validation_generator,
                    # callbacks=[early_stopping, save_best_model],
                    #callbacks=[save_best_model],
                    validation_steps=(nb_validation_samples // batch_size),
                    verbose=1
                   )

# output history
res = json.dumps(history.history)
fp = open('vgg_train_data.dat','w')
fp.write(res)
fp.close()

# Save model
model.save_weights('vgg_full_model_weights.h5')
model.save('vgg_model.h5')
