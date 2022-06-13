# -*- coding: utf-8 -*-
"""
Created on Wed May  8 14:10:24 2019

@author: iremn
"""


import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tflearn.data_utils import image_preloader
import numpy as np


X, Y = image_preloader('dataset', image_shape=(256, 256), mode='folder', categorical_labels=True, normalize=True, files_extension = ['.jpg', '.jpeg', '.png',".JPG"])
x = np.array(X)
y = np.array(Y)
train_images, test_images, train_labels, test_labels = train_test_split(x, y, train_size=0.9, test_size=0.1) 

from keras.utils import to_categorical
print('Eğitim verisinin şekli : ', train_images.shape, train_labels.shape)
print('Test verisinin şekli : ', test_images.shape, test_labels.shape)


dimData = np.prod(train_images.shape[1:])
train_data = train_images.reshape(train_images.shape[0], dimData)
test_data = test_images.reshape(test_images.shape[0], dimData)


train_data = train_data.astype('float32')
test_data = test_data.astype('float32')

train_data /= 255
test_data /= 255

from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(dimData,)))
model.add(Dense(512, activation='relu'))
model.add(Dense(train_labels.shape[1], activation='softmax'))

print(dimData)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_data, train_labels, batch_size=256, epochs=50, verbose=1, 
                   validation_data=(test_data, test_labels))

model.save('bitirmeModel.h5')
[test_loss, test_acc] =model.evaluate(test_data,test_labels)
print("Test verilerinde değerlendirme sonucu : Kayıp = {}, Dogruluk {}".format(test_loss, test_acc))


plt.subplot(121)
plt.plot(history.history['loss'], 'r')
plt.plot(history.history['val_loss'], 'b')
plt.legend(['Egitim Kayıbı', 'Dogrulama Kayıbı'])
plt.xlabel('Epochs ')
plt.ylabel('Kayıp')
plt.title('Kayıp Egrisi')


plt.subplot(122)
plt.plot(history.history['acc'], 'r')
plt.plot(history.history['val_acc'], 'b')
plt.legend(['Egitim Dogrulugu', 'Dogrulama Dogrulugu'])
plt.xlabel('Epochs ')
plt.ylabel('Dogruluk')
plt.title('Dogruluk Egrisi')

plt.show()



plt.figure()
plt.title('Test Edilen Kişi')
plt.subplot()
plt.imshow(test_images[1, :,:], cmap='gray')

tahmin=int(model.predict_classes(test_data[[1],:]))

#modelimizin tahmini görmek için csv de yazılan idye göre çekme işlemi..
import numpy as np
import pandas as pd
df = pd.read_csv("isimler.csv")
print("Tahmin edilen kişi:")
print((df['first_name'][tahmin]))
