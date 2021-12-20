#!/usr/bin/env python
# coding: utf-8

# In[6]:


# GPU 사용여부 확인
# from tensorflow.python.client import device_lib
# device_lib.list_local_devices()

import numpy as np
import pandas as pd
import os

dataDir = os.path.join(os.getcwd(),"xray_dataset_covid19");

for dirname, _, filenames in os.walk(dataDir):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[7]:


import struct
import numpy as np
import tensorflow as tf
from keras.layers import Conv2D
from keras.layers import Input
from keras.layers import BatchNormalization
from keras.layers import LeakyReLU
from keras.layers import Flatten,Dense,Dropout
from keras.layers import UpSampling2D,Activation,LeakyReLU,MaxPooling2D,Normalization,SeparableConv2D
from keras.layers.merge import add, concatenate
from keras.models import Model,Sequential
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
import keras.losses
import matplotlib.pyplot as plt
import cv2
from keras.preprocessing import image as im
from tensorflow.keras.optimizers import Adam,RMSprop,SGD


# In[8]:


train = os.path.join(dataDir,"train")
test = os.path.join(dataDir,"test")
print(train, test)


# In[9]:


from keras.preprocessing.image import ImageDataGenerator

new = ImageDataGenerator(rescale = 1.0/255.0,
                        horizontal_flip = True,
                        zoom_range = 0.2,
                        shear_range = 0.2,
                        width_shift_range = 0.01,
                        height_shift_range = 0.01)

train = new.flow_from_directory(train,
                                target_size = (224,224),
                                class_mode = 'binary',
                                color_mode = 'grayscale',
                                batch_size = 32)

valid = new.flow_from_directory(test,
                                target_size = (224,224),
                                class_mode = 'binary',
                                color_mode = 'grayscale',
                                batch_size = 32)

w_initializer = 'he_uniform'
f_activation = 'LeakyReLU'


# In[5]:


# CNN model (VGG16)
model1 = Sequential()
model1.add(Conv2D(64, (3,3), kernel_initializer=w_initializer,activation=f_activation, input_shape=(224,224,1),
                  padding='same'))
model1.add(Conv2D(64,kernel_size=(3,3), strides=(2,2),kernel_initializer=w_initializer,activation=f_activation,
                  padding='same'))
model1.add(MaxPooling2D(strides=(2,2)))
model1.add(Conv2D(128,kernel_size=(3,3),kernel_initializer=w_initializer,activation=f_activation,padding='same'))
model1.add(Conv2D(128,kernel_size=(3,3),kernel_initializer=w_initializer,activation=f_activation,padding='same'))
model1.add(MaxPooling2D(strides=(2,2)))
model1.add(Conv2D(256,kernel_size=(3,3),kernel_initializer=w_initializer,activation=f_activation,padding='same'))
model1.add(Conv2D(256,kernel_size=(3,3),kernel_initializer=w_initializer,activation=f_activation,padding='same'))
model1.add(Conv2D(256,kernel_size=(3,3),kernel_initializer=w_initializer,activation=f_activation,padding='same'))
model1.add(MaxPooling2D(strides=(2,2)))
model1.add(Conv2D(512,kernel_size=(3,3),kernel_initializer=w_initializer,activation=f_activation,padding='same'))
model1.add(Conv2D(512,kernel_size=(3,3),kernel_initializer=w_initializer,activation=f_activation,padding='same'))
model1.add(Conv2D(512,kernel_size=(3,3),kernel_initializer=w_initializer,activation=f_activation,padding='same'))
model1.add(MaxPooling2D(strides=(2,2)))

model1.add(Flatten())
model1.add(Dense(4096,activation=f_activation))
model1.add(Dense(4096,activation=f_activation))
model1.add(Dense(1,activation='sigmoid'))

model1.compile(optimizer = SGD(.001),loss='binary_crossentropy',metrics=['accuracy'])
model1.summary()

dot_res = model_to_dot(model1,show_shapes=True, dpi=50).create(prog='dot', format='svg')
fi = open("model1_visualization.svg", 'wb')
fi.write(dot_res)
fi.close()


# In[6]:


hist1 = model1.fit(train,validation_data=valid,epochs=10,batch_size=1)


# In[22]:


evaluate1 = model1.evaluate(valid)
print(evaluate1[1])


# In[8]:


plt.figure(1, figsize = (15, 5))
plt.subplot(1,2,1)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.plot( hist1.history["loss"], label = "Training Loss")
plt.plot( hist1.history["val_loss"], label = "Validation Loss")
plt.grid(True)
plt.legend()

plt.subplot(1,2,2)
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.plot( hist1.history["accuracy"], label = "Training Accuracy")
plt.plot( hist1.history["val_accuracy"], label = "Validation Accuracy")
plt.grid(True)
plt.legend()
plt.show()


# In[9]:


for i in os.listdir(os.path.join(dataDir,'test','PNEUMONIA')):
  locP = im.load_img(test+'/PNEUMONIA/'+i, target_size = (224,224), color_mode = 'grayscale')
  imagePneu = im.img_to_array(locP)
  imagePneu = tf.expand_dims(imagePneu, axis =0 )
  predPneumonia =model1.predict(imagePneu) 
for i in os.listdir(os.path.join(dataDir,'test','NORMAL')):
  locN = im.load_img(test+'/NORMAL/'+i, target_size = (224,224), color_mode = 'grayscale')
  imageNor = im.img_to_array(locN)
  imageNor = tf.expand_dims(imageNor, axis =0 )
  predNormal =model1.predict(imageNor)


# In[10]:


wrongPneu=[]
for i in range(len(predPneumonia)):
  if predPneumonia != 1:
    wrongPneu.append(i)

wrongNor=[]
for i in range(len(predNormal)):
  if predNormal != 0:
    wrongNor.append(i)
    
print(len(wrongPneu),len(wrongNor))


# In[11]:


locP = []
locN = []
for i in os.listdir(os.path.join(dataDir,'test','PNEUMONIA')):
  locP.append(im.load_img(test+'/PNEUMONIA/'+i, target_size = (224,224)))
for i in os.listdir(os.path.join(dataDir,'test','NORMAL')):
  locN.append(im.load_img(test+'/NORMAL/'+i, target_size = (224,224)))


# In[12]:


import matplotlib.pyplot as plt
plt.figure(figsize=(30,30))
for i in enumerate(locN):
    if (i[0]+1 < 21):
        plt.subplot(4,5,i[0]+1)
        plt.imshow(i[1])
        plt.xlabel('NORMAL')


# In[13]:


plt.figure(figsize=(30,30))
for i in enumerate(locP):
    if (i[0]+1 < 21):
        plt.subplot(4,5,i[0]+1)
        plt.imshow(i[1])
        plt.xlabel('PNEUMONIA')


# In[ ]:




