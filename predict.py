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




######### Loading the model ###############
model = load_model('binary.h5')
model.summary()


########### Predicting the model from the class labels reading every data from folder ############

num_of_classes = 2


listofdir = os.listdir('meerut-19')
number_of_files = len(listofdir)
totclass = [0 for i in range(num_of_classes)]
totmean = np.zeros((1,num_of_classes))
for i in range(number_of_files):
    imgpath = 'meerut-19'+ '/'+ str(i)+ '.png'
    print(imgpath)
    img = cv2.imread(imgpath)
    img = cv2.resize(img,(64,64))
    img = np.reshape(img,[1,64,64,3])
    img = preprocess_input(img)
    classes = model.predict(img)
    totmean += classes
    totclass[np.argmax(classes)] += 1
print(totclass)




totclass
print((totmean/number_of_files)*100)
