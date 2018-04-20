#coding=utf8
from keras import backend as K
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout, Activation
from keras.optimizers import rmsprop
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
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
batch_size = 128

train_data_dir = 'data/train'
validation_data_dir = 'data/validation'

nb_train_samples = 120000
nb_validation_samples = 42000

if K.image_data_format() == 'channels_first':
    input_shape = (3, image_size, image_size)
else:
    input_shape = (image_size, image_size, 3)

# Specify model

# callbacks
# early_stopping = EarlyStopping(monitor='val_loss', patience=3)
#save_best_model = ModelCheckpoint(filepath='model_.{epoch:02d}_{val_loss:.2f}.hdf5', verbose=1,
#        monitor='val_loss')

# instantiate Sequential model
model = Sequential()

# 定义第一层, 由于是回归模型, 因此只有一层
#model.add(Dense(units = 1, input_dim=2))
#model.add(Dense(units = 2, input_shape=input_shape, activation='softmax'))
model.add(Dense(512, input_shape=(image_size*image_size*3,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))

#model.add(Dense(units = 128, input_shape=(image_size, image_size, 3), activity_regularizer=regularizers.l1(0.001)))
#model.add(Flatten())
model.add(Dense(512))
model.add(Activation("relu"))
model.add(Dropout(0.5))

model.add(Dense(num_classes))
model.add(Activation('softmax'))
'''
model.add(Conv2D(filters=64, kernel_size=2, strides=2, activation='elu', kernel_initializer='glorot_normal', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=2, padding='same'))

model.add(Conv2D(filters=128, kernel_size=2, strides=2, activation='elu', kernel_initializer='glorot_normal'))
model.add(MaxPooling2D(pool_size=2, padding='same'))

model.add(Conv2D(filters=256, kernel_size=2, strides=2, activation='elu', kernel_initializer='glorot_normal'))
model.add(MaxPooling2D(pool_size=2, padding='same'))

model.add(Conv2D(filters=512, kernel_size=2, strides=2, activation='elu', kernel_initializer='glorot_normal'))
model.add(MaxPooling2D(pool_size=2, padding='same'))

model.add(Flatten())
model.add(Dense(128))
    #kernel_regularizer=regularizers.l2(0.001)))
                   # activity_regularizer=regularizers.l1(0.00001)))

model.add(Activation('elu'))
model.add(Dropout(0.5))

model.add(Dense(num_classes))
model.add(Activation('softmax'))
opt = rmsprop()
'''

model.summary()

#model.compile(loss = 'mse', optimizer = 'sgd')
#model.compile(loss = 'categorical_crossentropy', optimizer = 'sgd')
#model.compile(loss='sparse_categorical_crossentropy',
model.compile(loss='categorical_crossentropy',
                optimizer=rmsprop(),
                metrics=['accuracy'])


'''
model.compile(loss='categorical_crossentropy',
             optimizer = opt,
             metrics = ['accuracy'])
'''

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
history = model.fit_generator(generate_arrays_from_file(train_generator),
                    steps_per_epoch=(nb_train_samples // batch_size),
                    epochs=nb_epoch,
                    validation_data=generate_arrays_from_file(validation_generator),
                    # callbacks=[early_stopping, save_best_model],
                    #callbacks=[save_best_model],
                    validation_steps=(nb_validation_samples // batch_size),
                    verbose=1
                   )

# output history
res = json.dumps(history.history)
fp = open('mlp_train_data.dat','w')
fp.write(res)
fp.close()

# Save model
model.save_weights('mlp_full_model_weights.h5')
model.save('mlp_model.h5')
