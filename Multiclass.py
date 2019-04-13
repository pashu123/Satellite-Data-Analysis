## Multiclass Resnet 50 model
## Authors
## Prashant Kumar
## HariOm Ahlawat




import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from keras.layers import Dropout,Flatten
from keras.layers.convolutional import Conv2D,MaxPooling2D
from sklearn.cross_validation import train_test_split
from sklearn.utils import shuffle
from keras.utils.np_utils import to_categorical
from keras.preprocessing import image
import pickle
import pandas as pd
import random
import os
import cv2
from keras.layers import Input
from keras.applications.resnet50 import ResNet50,preprocess_input
from keras.models import Model
from keras.models import load_model





PATH = os.getcwd()
data_path = PATH + '/data'
data_dir_list = os.listdir(data_path)

############### Set no. of channels = 3 for color #########
img_rows = 64
img_cols = 64

## Define the number of classes ######
num_classes  = 10
labels_name = {'AnnualCrop':0,'Forest':1,'HerbaceousVegetation':2,'Highway':3,'Industrial':4,
            'Pasture':5,'PermanentCrop':6,'Residential':7,'River':8,'SeaLake':9}

img_data_list = []
labels_list = []

for dataset in data_dir_list:
    img_list=os.listdir(data_path+'/'+ dataset)
    print ('Loading the images of dataset-'+'{}\n'.format(dataset))
    label = labels_name[dataset]
    for img in img_list:
        img_path = data_path + '/'+ dataset + '/'+ img 
        img = image.load_img(img_path,target_size=(64,64))
        x = image.img_to_array(img)
        x = np.expand_dims(x,axis = 0)
        x = preprocess_input(x)
        img_data_list.append(x)
        labels_list.append(label)



########### Preprocessing the input ############

img_data = np.array(img_data_list)
img_data = img_data.astype('float32')
print (img_data.shape)
img_data = np.rollaxis(img_data,1,0)
print(img_data.shape)
img_data = img_data[0]
print(img_data.shape)


## Creating the one hot encoding and shuffling the data #########
labels = np.array(labels_list)
Y = to_categorical(labels,num_classes)
x,y = shuffle(img_data,Y)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1)
print(X_train.shape)
print(X_test.shape)


## Checking all the images are in right shape
assert(X_train.shape[0] == y_train.shape[0]),'The number of images is not equal to the number of labels'
assert(X_test.shape[0] == y_test.shape[0]),'The number of images is not equal to the number of labels'
assert(X_train.shape[1:] == (64,64,3)),'The dimensions of images are not (32,32,3)'
assert(X_test.shape[1:] == (64,64,3)),'The dimensions of images are not (32,32,3)'



##########  Augmentation Technique for generating the data ##########

# from keras.preprocessing.image import ImageDataGenerator
# datagen = ImageDataGenerator(width_shift_range = 0.1,
#                   height_shift_range = 0.1,
#                   zoom_range = 0.2,
#                   shear_range = 0.1,
#                   rotation_range = 10)
# datagen.fit(X_train)



# batches = datagen.flow(X_train,y_train,batch_size = 20)
# X_batch,y_batch = next(batches)
# fig,axes = plt.subplots(1,15,figsize = (20,5))
# fig.tight_layout()
# for i in range(15):
#   axes[i].imshow(X_batch[i].reshape(64,64,3))
#   axes[i].axis('off')






######### Resnet 50 model #################

image_input = Input(shape = (64,64,3))
model = ResNet50(input_tensor = image_input, include_top = False,weights = 'imagenet')
model.summary()

## Adding the last layer to the resnet50 model and finally compiling the model
last_layer = model.get_layer('activation_49').output
x= Flatten(name='flatten')(last_layer)
out = Dense(num_classes, activation='softmax', name='output_layer')(x)
custom_resnet_model = Model(inputs = image_input,outputs = out)
custom_resnet_model.summary()
custom_resnet_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])




# Without augmentation
hist = custom_resnet_model.fit(X_train, y_train, batch_size=32, epochs=1, verbose=1, validation_data=(X_test, y_test))


## With augmentation
# custom_resnet_model.fit_generator(datagen.flow(X_train,y_train,batch_size = 50),steps_per_epoch = 2000,epochs = 10,validation_data =(X_test,y_test),shuffle = 1)


custom_resnet_model.save('multi.h5')