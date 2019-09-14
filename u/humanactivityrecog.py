import keras
import numpy as np
#from parse import load_data
from tqdm import tqdm
import os
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import tflearn
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Activation, Dropout, Input
from keras.regularizers import l2
from tflearn.layers.conv import conv_2d,max_pool_2d
from tflearn.layers.core import input_data,dropout,fully_connected
from tflearn.layers.estimator import regression
from random import shuffle
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D, SeparableConv2D
from keras.layers import MaxPooling2D
from keras.layers.merge import add, concatenate
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import SpatialDropout2D
from keras.callbacks import ReduceLROnPlateau, CSVLogger, ModelCheckpoint
from keras.models import Sequential
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.models import Model, Sequential, load_model
from sklearn.preprocessing import LabelEncoder
from keras.models import load_model
from keras.layers import GlobalMaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers import Activation
count = 0
import os
#os.mkdir('hi')
num_classes=101
x=open('classInd.txt','r')
x1=x.read()
x2=list(x1.split('\n'))
x4=[]
for i in range(101):
    x3=x2[i].split(' ')[-1]
    x4.append(x3)
x5=[]
for i in range(101):
    x5.append(x4[i])
    print(x4)
train1_data = r'E:\Nikhil\python\ucf1\train'
validation1_data=r'C:\Users\Windows\Documents\action\humanactivity\val'
IMG_SIZE=160
IMG_SIZE1=160
LR = 1e-3

input = Input(shape=(160, 160, 3))

# Block 1
layer0 = Conv2D(32, (7, 7), padding='same', kernel_regularizer=l2(1e-4), use_bias=False, kernel_initializer='he_normal', name='sep_conv1')(input) # 7rf ji=1
layer0 = BatchNormalization(name='bn1')(layer0)
layer0 = Activation('relu', name='relu1')(layer0)
layer0 = MaxPooling2D(pool_size=(2,2), strides=(2,2), name='mp1')(layer0) # 8rf ji=1 jo=2

skip_connection_1 = layer0


# Block 2
layer1 = Conv2D(48, (3, 3), padding='same', kernel_regularizer=l2(1e-4), use_bias=False, kernel_initializer='he_normal', name='sep_conv2')(layer0) # 12rf ji=2
layer1 = BatchNormalization(name='bn2')(layer1)
layer1 = Activation('relu', name='relu2')(layer1)

layer2 = Conv2D(48, (3, 3), padding='same', kernel_regularizer=l2(1e-4), use_bias=False, kernel_initializer='he_normal', name='sep_conv3')(layer1) # 16rf ji=2
layer2 = BatchNormalization(name='bn3')(layer2)
layer2 = Activation('relu', name='relu3')(layer2)

layer3 = Conv2D(48, (3, 3), padding='same', kernel_regularizer=l2(1e-4), use_bias=False, kernel_initializer='he_normal', name='sep_conv4')(layer2) # 20rf ji=2
layer3 = BatchNormalization(name='bn4')(layer3)
layer3 = Activation('relu', name='relu4')(layer3)

layer4 = Conv2D(48, (3, 3), padding='same', kernel_regularizer=l2(1e-4), use_bias=False, kernel_initializer='he_normal', name='sep_conv5')(layer3) # 24rf ji=2
layer4 = BatchNormalization(name='bn5')(layer4)
layer4 = Activation('relu', name='relu5')(layer4)

layer5 = concatenate([skip_connection_1, layer4])
layer5 = Conv2D(48, (1, 1), padding='same', kernel_regularizer=l2(1e-4), use_bias=False, kernel_initializer='he_normal', name='sep_conv6')(layer5) # 28rf ji=2
layer5 = BatchNormalization(name='bn6')(layer5)
layer5 = Activation('relu', name='relu6')(layer5)
layer5 = MaxPooling2D(pool_size=(2, 2), strides=(2,2), name='mp2')(layer5) # 30rf ji=2 jo=4

skip_connection_2 = layer5


# Block 3
layer6 = Conv2D(64, (3, 3), padding='same', kernel_regularizer=l2(1e-4), use_bias=False, kernel_initializer='he_normal', name='sep_conv7')(layer5) # 38rf ji=4
layer6 = BatchNormalization(name='bn7')(layer6)
layer6 = Activation('relu', name='relu7')(layer6)

layer7 = Conv2D(64, (3, 3), padding='same', kernel_regularizer=l2(1e-4), use_bias=False, kernel_initializer='he_normal', name='sep_conv8')(layer6) # 46rf ji=4
layer7 = BatchNormalization(name='bn8')(layer7)
layer7 = Activation('relu', name='relu8')(layer7)

layer8 = Conv2D(64, (3, 3), padding='same', kernel_regularizer=l2(1e-4), use_bias=False, kernel_initializer='he_normal', name='sep_conv9')(layer7) # 54rf ji=4
layer8 = BatchNormalization(name='bn9')(layer8)
layer8 = Activation('relu', name='relu9')(layer8)

layer9 = Conv2D(64, (3, 3), padding='same', kernel_regularizer=l2(1e-4), use_bias=False, kernel_initializer='he_normal', name='sep_conv10')(layer8) # 62rf ji=4
layer9 = BatchNormalization(name='bn10')(layer9)
layer9 = Activation('relu', name='relu10')(layer9)

layer10 = concatenate([skip_connection_2, layer9])
layer10 = Conv2D(64, (1,1), padding='same', kernel_regularizer=l2(1e-4), use_bias=False, kernel_initializer='he_normal', name='sep_conv11')(layer10) # 70rf ji=4
layer10 = BatchNormalization(name='bn11')(layer10)
layer10 = Activation('relu', name='relu11')(layer10)
layer10 = MaxPooling2D(pool_size=(2, 2), strides=(2,2), name='mp3')(layer10) # 74rf ji=4 jo=8

skip_connection_3 = layer10


# Block 4



# Output block
layer21 = Conv2D(num_classes, (1, 1), kernel_regularizer=l2(1e-4), use_bias=False, kernel_initializer='he_normal', name='sep_conv22')(layer10) # 402rf ji=32
layer21 = GlobalAveragePooling2D(name='gap1')(layer21)


output = Activation('softmax', name='softmax1')(layer21)

epochs1=50
lrate=0.01
decay=lrate/epochs1
sgd = SGD(lr=lrate,momentum=0.9,decay=decay,nesterov=False)
model = Model(inputs=[input], outputs=[output])
model.summary()

model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])
print(model.summary()) 
filepath="weights-improvement-{epoch:02d}-{acc:.2f}.hdf5"
model_checkpoint = ModelCheckpoint(filepath, monitor='acc', verbose=1, save_best_only=True, mode='max')
callbacks_list=[model_checkpoint]
#model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=epochs1,batch_size=16,verbose=2)
#scores = model.evaluate(X_test,y_test,verbose=0)
#print("Accuracy:%.2f%%"%(scores[1]*100))


train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1. / 255)
train_generator = train_datagen.flow_from_directory(
    train1_data,
    target_size=(IMG_SIZE, IMG_SIZE1),
    classes=x5,
    batch_size=16
    
    )

validation_generator = test_datagen.flow_from_directory(
    validation1_data,
    target_size=(IMG_SIZE,IMG_SIZE1),
    classes=['boxsng','handclapping','handwaving','walking'],
    batch_size=64,
    )

model.fit_generator(
    
    train_generator,
    epochs=15,
     #nb_train_samples//batch_size
    steps_per_epoch=68000/16,
    callbacks = callbacks_list
    
    )

model.save('model3-imggen.h5')
