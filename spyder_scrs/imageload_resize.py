# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 17:43:22 2017

@author: atpandey
"""
#%%

import cv2
import numpy as np
import matplotlib.pyplot as plt
#load all pictures as grayscale
#1. load yield pic
yield_pic=cv2.imread('../yield.jpg',0)
print("yiledleft has shape",yield_pic.shape)
#plt.imshow(yield_pic,cmap='gray')

yield_pic_res = cv2.resize(yield_pic,(32, 32), interpolation = cv2.INTER_CUBIC)
#cv2.imwrite('yield_pic_res.png',yield_pic_res)
#plt.imshow(yield_pic_res,cmap='gray')

#2. curve left
curve_left=cv2.imread('../curveleft.jpg',0)
print("curve_left has shape",curve_left.shape)
#plt.imshow(curve_left,cmap='gray')
curve_left_res = cv2.resize(curve_left,(32, 32), interpolation = cv2.INTER_CUBIC)

#cv2.imwrite('curve_left_res.png',curve_left_res)
#plt.imshow(curve_left_res,cmap='gray')

#3. priority road
priority_road=cv2.imread('../priorityroad.jpg',0)
print("priority_road has shape",priority_road.shape)
#plt.imshow(priority_road,cmap='gray')
priority_road_res = cv2.resize(priority_road,(32, 32), interpolation = cv2.INTER_CUBIC)

#cv2.imwrite('priority_road_res.png',priority_road_res)
#plt.imshow(priority_road_res,cmap='gray')



#4. Road work
road_work=cv2.imread('../construction.jpg',0)
print("road_work has shape",road_work.shape)
#plt.imshow(road_work,cmap='gray')
road_work_res = cv2.resize(road_work,(32, 32), interpolation = cv2.INTER_CUBIC)

#cv2.imwrite('road_work_res.png',road_work_res)
#plt.imshow(road_work_res,cmap='gray')
#5. speed limit 50
speed_50=cv2.imread('../50speed.jpg',0)
print("speed_50 has shape",speed_50.shape)
#plt.imshow(speed_50,cmap='gray')
speed_50_res = cv2.resize(speed_50,(32, 32), interpolation = cv2.INTER_CUBIC)

#cv2.imwrite('speed_50_res.png',speed_50_res)
#plt.imshow(speed_50_res,cmap='gray')


#6. speed limit 30
speed_30=cv2.imread('../30speedlimit.jpg',0)
print("speed_30 has shape",speed_30.shape)
#plt.imshow(speed_30,cmap='gray')
speed_30_res = cv2.resize(speed_30,(32, 32), interpolation = cv2.INTER_CUBIC)

#cv2.imwrite('speed_30_res.png',speed_50_res)
#plt.imshow(speed_30_res,cmap='gray')



fig_t, axs_t = plt.subplots(2,3, figsize=(8, 8))
axs_t = axs_t.ravel()
fig_r, axs_r = plt.subplots(2,3, figsize=(8, 8))
axs_r = axs_r.ravel()

#%%
import cv2
import numpy as np
import matplotlib.pyplot as plt
#load colored images
yield_pic=plt.imread('../yield.jpg')
curve_left=plt.imread('../curveleft.jpg')
priority_road=plt.imread('../priorityroad.jpg')
road_work=plt.imread('../construction.jpg')
speed_50=plt.imread('../50speed.jpg')
speed_30=plt.imread('../30speedlimit.jpg')

fig_c, axs_c = plt.subplots(2,3, figsize=(8, 8))
axs_c = axs_c.ravel()
#plot colored pics
axs_c[0].imshow(yield_pic)
axs_c[1].imshow(curve_left)
axs_c[2].imshow(priority_road)
axs_c[3].imshow(road_work)
axs_c[4].imshow(speed_50)
axs_c[5].imshow(speed_30)

#now convert them to grayscale
yield_pic_g=cv2.cvtColor(yield_pic,cv2.COLOR_BGR2GRAY)
curve_left_g=cv2.cvtColor(curve_left,cv2.COLOR_BGR2GRAY)
priority_road_g=cv2.cvtColor(priority_road,cv2.COLOR_BGR2GRAY)
road_work_g=cv2.cvtColor(road_work,cv2.COLOR_BGR2GRAY)
speed_50_g=cv2.cvtColor(speed_50,cv2.COLOR_BGR2GRAY)
speed_30_g=cv2.cvtColor(speed_30,cv2.COLOR_BGR2GRAY)


fig_g, axs_g = plt.subplots(2,3, figsize=(8, 8))
axs_g = axs_g.ravel()
#plot colored pics
axs_g[0].imshow(yield_pic_g,cmap='gray')
axs_g[1].imshow(curve_left_g,cmap='gray')
axs_g[2].imshow(priority_road_g,cmap='gray')
axs_g[3].imshow(road_work_g,cmap='gray')
axs_g[4].imshow(speed_50_g,cmap='gray')
axs_g[5].imshow(speed_30_g,cmap='gray')

#now resize to 32x32
yield_pic_res = cv2.resize(yield_pic_g,(32, 32), interpolation = cv2.INTER_CUBIC)
curve_left_res=cv2.resize(curve_left_g,(32, 32), interpolation = cv2.INTER_CUBIC)
priority_road_res=cv2.resize(priority_road_g,(32, 32), interpolation = cv2.INTER_CUBIC)
road_work_res=cv2.resize(road_work_g,(32, 32), interpolation = cv2.INTER_CUBIC)
speed_50_res=cv2.resize(speed_50_g,(32, 32), interpolation = cv2.INTER_CUBIC)
speed_30_res=cv2.resize(speed_30_g,(32, 32), interpolation = cv2.INTER_CUBIC)



fig_r, axs_r = plt.subplots(2,3, figsize=(8, 8))
axs_r = axs_r.ravel()
axs_r[0].imshow(yield_pic_res,cmap='gray')
axs_r[1].imshow(curve_left_res,cmap='gray')
axs_r[2].imshow(priority_road_res,cmap='gray')
axs_r[3].imshow(road_work_res,cmap='gray')
axs_r[4].imshow(speed_50_res,cmap='gray')
axs_r[5].imshow(speed_30_res,cmap='gray')

#%%
speed_30=plt.imread('../30speedlimit_n.jpg')
#plt.imshow(speed_30)
speed_30_res=cv2.resize(speed_30,(32, 32), interpolation = cv2.INTER_AREA)
plt.imshow(speed_30_res)

#%%
import tensorflow as tf
#fname='../30speedlimit_n.jpg'
#speed_30=tf.image.decode_jpeg(fname)
speed_30=plt.imread('../30speedlimit_n.jpg')
w=speed_30.shape[1]
h=speed_30.shape[0]
ch=speed_30.shape[2]
print(h,w,ch)
#speed_30_res=tf.image.resize_images(speed_30, [32, 32])
speed_30_res=tf.image.resize_area(speed_30.reshape(-1,h,w,ch), [32, 32])
print(speed_30_res.shape)
with tf.Session() as sess:
    img=sess.run(speed_30_res)
#    print(img.shape)
plt.imshow(img.squeeze(),cmap='gray')

#%%
from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean

speed_30=plt.imread('../30speedlimit_n.jpg')
#image_rescaled = rescale(image, 1.0 / 4.0, anti_aliasing=False)
image_resized = resize(speed_30, (32, 32))
plt.imshow(image_resized)

