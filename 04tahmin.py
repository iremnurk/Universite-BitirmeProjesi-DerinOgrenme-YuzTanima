# -*- coding: utf-8 -*-
"""
Created on Fri May 10 15:26:26 2019

@author: iremn
"""
import matplotlib.pyplot as plt

from keras.models import load_model
model=load_model('bitirmeModel.h5')
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])


#dışarıdan girilen test verisi ile tahmin sonucunu görme

from keras.preprocessing import image
import numpy as np

img = image.load_img(path="test/nurseda.jpg",target_size=(256,256,3))
img = image.img_to_array(img)
test_img = img.reshape((1,196608))
img_class = model.predict_classes(test_img)
prediction = img_class[0]

import numpy as np
import pandas as pd
df = pd.read_csv("isimler.csv")
print((df['first_name'][prediction]))
tahminisim=df['first_name'][prediction]

import cv2
resim=cv2.imread('test/nurseda.jpg')
resim= cv2.cvtColor(resim,cv2.COLOR_RGB2BGR)
#cv2.imshow('Tahmin',resim)
plt.figure()
plt.title('Tahmin Edilen Kişi= %s' % (tahminisim))
plt.subplot()
plt.imshow(resim)

