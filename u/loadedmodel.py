
from keras.models import load_model
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator

import numpy as np
from keras.optimizers import SGD

import os
import pandas as pd

num_classes=10
x=open('classInd.txt','r')
x1=x.read()
x2=list(x1.split('\n'))
x4=[]
for i in range(101):
    x3=x2[i].split(' ')[-1]
    x4.append(x3)
x5=[]
for i in range(10):
    x5.append(x4[i])
    print(x5)
#model = load_model('model-accuracy-99%.h5')
test_dir = r'E:\Nikhil\python\videoclass\videoclass\test'
test1_dir = r'C:\Users\Windows\Documents\action\humanactivity\test'
test2_dir=r'E:\Nikhil\python\ucf1\test'
test3_dir=r'E:\Nikhil\python\machinelearningex\videoclass\videoclass\test'
test4_dir=r'E:\Nikhil\python'
ucf_dir=r'E:\Nikhil\python\ucf1\fake'
train1_data=r'C:\Users\Windows\Documents\action\humanactivity\train'
validation1_data=r'C:\Users\Windows\Documents\action\humanactivity\val'
model=Sequential()
model=load_model('model10ucf.hdf5')
lrate=0.01
epochs1=50
decay=lrate/epochs1
sgd = SGD(lr=lrate,momentum=0.9,decay=decay,nesterov=False)
batch_size=64
model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])
test_datagen= ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    
    ucf_dir,
    classes=['test'],
    target_size=(224,224),
    
    shuffle=False,
    batch_size=10,
    )

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)
train_generator = train_datagen.flow_from_directory(
    test2_dir,
    target_size=(224,224),
    batch_size=16,
    shuffle=False,
    classes=x5,
    )
    
test_generator.reset()
#train_generator.reset()
#print(test_generator.filenames)
#model1=model.predict_generator(test_generator,1200)
test_datagen = ImageDataGenerator(rescale=1. / 255)
validation_generator = test_datagen.flow_from_directory(
    test2_dir,
    target_size=(224,224),
    classes=x5,
    batch_size=32,
    )

'''
model.fit_generator(test_datagen,steps_per_epoch=5,epochs=1,verbose=0)
scores=model.evaluate_generator(test_datagen,train_datagen)
print(scores[0])
'''



pred= model.predict_generator(test_generator, test_generator.n//batch_size+1)
predicted_class_indices=np.argmax(pred,axis=1)
labels = (validation_generator.class_indices)
labels2 = dict((v,k) for k,v in labels.items())
predictions = [labels2[k] for k in predicted_class_indices]
#print(predicted_class_indices)
#print (labels)
          
#print (predictions)

dt={}
for i in range(len(test_generator.filenames)):
        
    name=test_generator.filenames[i].split('\\')[-1]
    dt[name]=predictions[i]
#print(dt)

'''
df1= pd.DataFrame(columns=['images','predicted'])
df1=pd.DataFrame(dt,index=['id'])
print(df1)
df1.to_csv('images.csv')

count=0
df=pd.read_csv('test1.csv',usecols=['predicted_output'])
for i in range(53101):
    if predictions[i]==df['predicted_output'][i]:
        count=count+1
accuracy=count/len(predictions)
print(accuracy)
'''
    

    

