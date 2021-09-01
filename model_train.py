#importing the packages
import cv2                          
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Flatten,Conv2D,MaxPooling2D
import os

training_dir = '/datasets/train'
validation_dir = '/datasets/test'

train_data_gen = ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)
validation_data_gen = ImageDataGenerator(rescale= 1./255)

train_data_set = train_data_gen.flow_from_directory(training_dir,target_size=(48,48),color_mode='grayscale',batch_size=32,class_mode="categorical",shuffle=True)

validation_data_set = validation_data_gen.flow_from_directory(validation_dir,target_size=(48,48),batch_size=32,color_mode='grayscale',class_mode="categorical",shuffle=True)

# creating structure
model = Sequential()

model.add(Conv2D(16, kernel_size=(3, 3), activation='elu', input_shape=(48, 48, 1)))
model.add(Conv2D(32, kernel_size=(3, 3), activation='elu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, kernel_size=(3, 3), activation='elu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(3, 3), activation='elu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='elu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation='elu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer = "adam",metrics=['accuracy'])

# Training neural model
model_info = model.fit_generator(train_data_set,steps_per_epoch=28709 // 64,epochs=35, validation_data=validation_data_set,validation_steps=7178 // 64)

#save model structure in json & .h5 file
model_json = model.to_json()
with open("data_model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights('data_model.h5')