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
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D, SeparableConv2D
from tflearn.layers.core import input_data,dropout,fully_connected
from tflearn.layers.estimator import regression
from random import shuffle
from keras.layers.convolutional import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.callbacks import ReduceLROnPlateau, CSVLogger, ModelCheckpoint
from keras.layers import Flatten
from keras.layers import Dropout
from keras.models import Sequential
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from keras.models import load_model
from keras.layers import Activation


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
validation1_data=r'E:\\Nikhil\\python\\machinelearningex\\videoclass\\videoclass\\test\\'
IMG_SIZE=150
IMG_SIZE1=150
LR = 1e-3
'''
def create_label(image_name):
    word_label=image_name.split('_')[-2]
    if word_label=='boxing':
        return 'boxing'
    if word_label =='handclapping':
        return 'handclapping'
    if word_label =='handwaving':
        return 'handwaving'
    if word_label == 'jogging':
        return 'jogging'
    if word_label == 'running':
        return 'running'
    if word_label == 'walking':
        return 'walking'
    

def create_train_data():
    training_data = []
    for img in tqdm(os.listdir(train1_data)):
        path = os.path.join(train1_data,img)
        img_data = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        img_data=cv2.resize(img_data,(IMG_SIZE,IMG_SIZE1))
        training_data.append([np.array(img_data),create_label(img)])
    shuffle(training_data)
    np.save('train_data.npy',training_data)
    return training_data

def create_test_data():
    test_data = []
    for img in tqdm(os.listdir(validation1_data)):
        path = os.path.join(validation1_data,img)
        img_data = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        img_data=cv2.resize(img_data,(IMG_SIZE,IMG_SIZE1))
        test_data.append([np.array(img_data),create_label(img)])
    shuffle(test_data)
    np.save('test_data.npy',test_data)
    return test_data

train_data = create_train_data()
test_data=create_test_data()

train = train_data[:-1]
test = train_data[-1200:]

X_train = np.array([i[0] for i in train]).reshape(-1,IMG_SIZE,IMG_SIZE1,1)
y_train = [i[1] for i in train]
X_test = np.array([i[0] for i in test]).reshape(-1,IMG_SIZE,IMG_SIZE1,1)
y_test =[i[1] for i in test]

X_train=X_train.astype('float32')
X_test=X_test.astype('float32')
print(y_test)

encoder=LabelEncoder()
encoder.fit(y_test)
y_test=encoder.transform(y_test)
encoder.fit(y_train)
y_train=encoder.transform(y_train)
print(y_test)

X_train=X_train/255.0
X_test=X_test/255.0
y_train=np_utils.to_categorical(y_train)
y_test=np_utils.to_categorical(y_test)
num_classes=6
print(y_test)
'''
model=Sequential()
model.add(Conv2D(32,3,3,input_shape=(IMG_SIZE,IMG_SIZE,3),border_mode='same',
                        activation='relu'))
model.add(Dropout(0.2))
model.add(Conv2D(32,3,3,activation='relu',border_mode='same'
                        ))
model.add(MaxPooling2D(pool_size=(2,2),dim_ordering="th"))

model.add(Conv2D(64,3,3,activation='relu',border_mode='same'
                        ))
model.add(Dropout(0.2))
model.add(Conv2D(64,3,3,activation='relu',border_mode='same'))
model.add(MaxPooling2D(pool_size=(2,2),dim_ordering="th"))


model.add(Conv2D(128,3,3,activation='relu',border_mode='same'))
model.add(Dropout(0.2))
model.add(Conv2D(128,3,3,activation='relu',border_mode='same'))
model.add(MaxPooling2D(pool_size=(2,2),dim_ordering="th"))

model.add(Conv2D(101, (1, 1), use_bias=False, kernel_initializer='he_normal', name='sep_conv22'))
model.add(GlobalAveragePooling2D(name='gap1'))


model.add(Activation('softmax',name='softmax'))

epochs1=50
lrate=0.01
decay=lrate/epochs1
sgd = SGD(lr=lrate,momentum=0.9,decay=decay,nesterov=False)
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
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
    batch_size=16,
    classes=x5,
    )

validation_generator = test_datagen.flow_from_directory(
    validation1_data,
    target_size=(IMG_SIZE,IMG_SIZE1),
    classes=['boxing','handclapping','handwaving','jogging','running','walking'],
    batch_size=10,
    )

model.fit_generator(
    
    train_generator,
    epochs=15,
     #nb_train_samples//batch_size
    steps_per_epoch=68000/16,
    callbacks = callbacks_list
    )
    
model.save_weights('model1-imggen.h5')
