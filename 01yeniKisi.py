# -*- coding: utf-8 -*-
"""
Created on Wed May  8 14:09:48 2019

@author: iremn
"""


import cv2
import os


kisi_id=input('\n yeni kisinin idsini giriniz\n  :')
yenikisi=input('\n yeni kisinin adını giriniz\n  :')
os.chdir('dataset')
os.mkdir(str(yenikisi))

