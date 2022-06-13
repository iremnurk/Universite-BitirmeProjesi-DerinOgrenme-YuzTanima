# -*- coding: utf-8 -*-
"""
Created on Wed May  8 14:10:05 2019

@author: iremn
"""

import cv2

camera = cv2.VideoCapture(0)

for i in range(15):
    return_value, image = camera.read()
    
    outfile = 'dataset/%s/%s.%s.jpg' % (str(yenikisi),yenikisi,str(i))
   
    cv2.imwrite(outfile, image)
    cv2.imshow('Kamera',image)
    
   
    cv2.waitKey(250) 

del(camera)

cv2.destroyAllWindows()

#csv ye yeni eklenen kişinin yazılması
import csv
row = [kisi_id, yenikisi]

with open('ornek.csv', 'a') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerow(row)

csvFile.close()