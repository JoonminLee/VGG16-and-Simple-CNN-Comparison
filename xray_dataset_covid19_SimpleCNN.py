#!/usr/bin/env python
# coding: utf-8

# In[11]:


import os
import matplotlib.pyplot as plt
import numpy as np

dataDir = os.path.join(os.getcwd(),"xray_dataset_covid19")
trainDir = os.path.join(dataDir, "train")
testDir = os.path.join(dataDir, "test")

folders = os.listdir(trainDir)
print(folders)

image_data = []
labels = []

label_dict = {
    'PNEUMONIA' : 0,
    'NORMAL' : 1
}

from keras.preprocessing import image

for i in folders :
    path = os.path.join(trainDir,i)
    for j in os.listdir(path):
        img = image.load_img(os.path.join(path,j),target_size=(224,224))
        img_array = image.img_to_array(img)
        image_data.append(img_array)
        labels.append(label_dict[i])


# In[12]:


print(len(image_data),len(labels))


# In[13]:


import random
combined = list(zip(image_data,labels))
random.shuffle(combined)
image_data[:],labels[:]=zip(*combined)


# In[14]:


print(labels)


# In[15]:


x_train = np.array(image_data)
y_train = np.array(labels)

print(x_train.shape,y_train.shape)


# In[16]:


from keras.utils import np_utils

y_train = np_utils.to_categorical(y_train)
print(x_train.shape,y_train.shape)


# In[17]:


from keras.preprocessing.image import ImageDataGenerator
augment = ImageDataGenerator( 
                              rotation_range=20,
                              width_shift_range=0.01, 
                              height_shift_range=0.01, 
                              horizontal_flip=False, 
                              vertical_flip=False,
                            )
augment.fit(x_train)


# In[18]:


from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.regularizers import l2
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot


# In[19]:


# Simple CNN model
model = Sequential()
model.add(Conv2D(filters=16,kernel_size=(3,3),input_shape=(224,224,3),activation='relu',kernel_regularizer=l2(0.01)))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Conv2D(filters=32,kernel_size=(3,3),input_shape=(224,224,3),activation='relu',kernel_regularizer=l2(0.01)))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Conv2D(filters=64,kernel_size=(3,3),input_shape=(224,224,3),activation='relu',kernel_regularizer=l2(0.01)))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Conv2D(filters=64,kernel_size=(3,3),input_shape=(224,224,3),activation='relu',kernel_regularizer=l2(0.01)))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(224))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(2))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

filepath="covid_detection.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=2, save_best_only=True, mode='min',patience=3)
callbacks_list = [checkpoint]


# In[20]:


model.summary()

dot_res = model_to_dot(model,show_shapes=True, dpi=50).create(prog='dot', format='svg')
fi = open("model_visualization.svg", 'wb')
fi.write(dot_res)
fi.close()


# In[21]:


hist = model.fit(x_train,y_train,batch_size=32, epochs = 25, validation_split = 0.10,callbacks=callbacks_list)


# In[22]:


plt.figure(1, figsize = (15, 5))
plt.subplot(1,2,1)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.plot( hist.history["loss"], label = "Training Loss")
plt.plot( hist.history["val_loss"], label = "Validation Loss")
plt.grid(True)
plt.legend()

plt.subplot(1,2,2)
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.plot( hist.history["accuracy"], label = "Training Accuracy")
plt.plot( hist.history["val_accuracy"], label = "Validation Accuracy")
plt.grid(True)
plt.legend()
plt.show()


# In[23]:


test_image_data = []
test_labels = []

test_folders = os.listdir(testDir)
print(test_folders)

label_dict = {
    'PNEUMONIA':0,
    'NORMAL':1
}

from keras.preprocessing import image

for ix in test_folders:
    path = os.path.join(testDir,ix)
    for im in os.listdir(path):
        img = image.load_img(os.path.join(path,im),target_size = ((224,224)))
        img_array = image.img_to_array(img)
        test_image_data.append(img_array)
        test_labels.append(label_dict[ix])

combined = list(zip(test_image_data,test_labels))
test_image_data[:],test_labels[:] = zip(*combined)

x_test = np.array(test_image_data)
y_test = np.array(test_labels)

from keras.utils import np_utils

y_test = np_utils.to_categorical(y_test)
print(x_test.shape,y_test.shape)


# In[37]:


evaluate1 = model.evaluate(x_test, y_test)
print(evaluate1[1])


# In[25]:


from sklearn.metrics import classification_report,confusion_matrix
predictions = model.predict(x_test, batch_size = 32)
pred = np.argmax(predictions, axis=1)


# In[26]:


print(classification_report(test_labels, pred))


# In[27]:


print(confusion_matrix(test_labels, pred))


# In[ ]:




