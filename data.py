# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 19:34:49 2017

@author: ray
"""

import os
import glob
from PIL import Image
import numpy as np
import scipy.misc as im



path = '/home/ray/python_code/keras_examples/pix2pix/CMP_facade_DB_base/base'
new_path = '/home/ray/python_code/keras_examples/pix2pix/' 
resize_pic = 'resize_pic/'
resize_target = 'resize_tag/'
#all pics are named begin with 'cmp_bo'
pic_name = 'cmp_b0'
pic_nb = len(os.listdir(new_path + resize_pic))
def resize(path):
    if not os.path.exists(resize_pic):
        os.mkdir(resize_pic)
    if not os.path.exists(resize_target):
        os.mkdir(resize_target)
    pic = glob.glob(os.path.join(path,'*.jpg'))
    pic.sort()
    target = glob.glob(os.path.join(path,'*.png'))
    target.sort()
    count1 = 1
    count2 = 1
    for i in pic:
        #pic_name = os.path.split(i)[-1]
        img = Image.open(i)
        resize_img = img.resize((256,256),Image.ANTIALIAS)
        #Renamed -> easy load
        resize_img.save(resize_pic + '%d.jpg' % count1)
        count1 += 1
    for j in target:
        #tag_name = os.path.split(j)[-1]
        tag = Image.open(j)
        resize_tag = tag.resize((256,256),Image.ANTIALIAS)
        #Renamed -> easy load
        resize_tag.save(resize_target + '%d.png' % count2)
        count2 +=1
        
      
def load():
    pic = []
    target = []
    for i in range(pic_nb):
        img = im.imread(new_path + resize_pic + '%d.jpg' % (i+1))
        pic.append(img)
    for j in range(pic_nb):
        tag = im.imread(new_path + resize_target + '%d.png' % (j+1))
        target.append(tag)
    return np.array(pic),np.array(target)
    #random_number = np.random.randint(0,pic_nb-1,size=Batchsize)
    
    


        
if __name__ == '__main__':
    #resize(path)
    a,b = load()