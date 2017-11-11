# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 20:09:58 2017

@author: atpandey
"""

#%%
import random
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
def noisy(noise_typ,image):
  if noise_typ == "gauss":
      row,col,ch= image.shape
      mean = 0
      #var = 0.1
      var = 0.01
      sigma = var**0.5
      gauss = np.random.normal(mean,sigma,(row,col,ch))
      gauss = gauss.reshape(row,col,ch)
      noisy = image + gauss
      return noisy
  elif noise_typ == "s&p":
      row,col,ch = image.shape
      s_vs_p = 0.5
      amount = 0.004
      out = np.copy(image)
      # Salt mode
      num_salt = np.ceil(amount * image.size * s_vs_p)
      coords = [np.random.randint(0, i - 1, int(num_salt)) \
      for i in image.shape]
      out[coords] = 1
     
      # Pepper mode
      num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
      coords = [np.random.randint(0, i - 1, int(num_pepper)) \
      for i in image.shape]
      out[coords] = 0
      return out

  elif noise_typ == "poisson":
      vals = len(np.unique(image))
      vals = 2 ** np.ceil(np.log2(vals))
      noisy = np.random.poisson(image * vals) / float(vals)
      return noisy
  elif noise_typ =="speckle":
      row,col,ch = image.shape
      gauss = np.random.randn(row,col,ch)
      gauss = gauss.reshape(row,col,ch)        
      noisy = image + image * gauss
      return noisy


#%%
img=cv2.imread('lenet.png')
plt.figure(figsize=(15,15))
plt.imshow(img)

img_gauss=noisy('gauss',img)
plt.figure(1,figsize=(15,15))
plt.imshow(img_gauss)

img_sp=noisy('s&p',img)
plt.figure(2,figsize=(15,15))
plt.imshow(img_sp)

img_poisson=noisy('poisson',img)
plt.figure(3,figsize=(15,15))
plt.imshow(img_poisson)

img_speckle=noisy('speckle',img)
plt.figure(4,figsize=(15,15))
plt.imshow(img_speckle)

#%%

# Load pickled data
import pickle

# TODO: Fill this in based on where you saved the training and testing data

training_file = 'train.p'
validation_file='valid.p'
testing_file = 'test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']

#%%

### Replace each question mark with the appropriate value. 
### Use python, pandas or numpy methods rather than hard coding the results
import numpy as np
# TODO: Number of training examples
n_train = X_train.shape[0]

# TODO: Number of validation examples
n_validation = X_valid.shape[0]

# TODO: Number of testing examples.
n_test = X_test.shape[0]

# TODO: What's the shape of an traffic sign image?
image_shape = (X_train.shape[1],X_train.shape[1])

# TODO: How many unique classes/labels there are in the dataset.
n_classes = len(set(y_train))

print("Number of training examples =", n_train)
print("Number of validation examples =", n_validation)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)

#%%
import random
import matplotlib.pyplot as plt
# Visualizations will be shown in the notebook.
#%matplotlib inline

#plot a randomly selected image
index_train=random.randint(0, len(X_train))
print("Chosen Index in training set is:",index_train)
image_train = X_train[index_train]
plt.figure(figsize=(1,1))
plt.imshow(image_train)
print("Picture is of type:",y_train[index_train])

#count of each sign type
#n_classes = len(set(y_train))
plt.figure(2,figsize=(6,5))
count_signtypes={}
for stype in y_train:
    if stype not in count_signtypes:
        count_signtypes[stype]=1
    else:
        count_signtypes[stype] +=1

lists_t = sorted(count_signtypes.items()) # sorted by key, return a list of tuples

x_t, y_t = zip(*lists_t) # unpack a list of pairs into two tuples

plt.scatter(x_t, y_t)
#plt.bar(range(len(count_signtypes)),count_signtypes.values())
plt.bar(x_t,y_t)
plt.show()   
#print(count_signtypes[0],count_signtypes[1],count_signtypes[2],count_signtypes[3],count_signtypes[4],\
#      count_signtypes[5])
max_pics_train=np.argmax(y_t)
print("Max number of examples are of type:",x_t[max_pics_train])
#%%
cntr=0
ims_to_add=[]
for j in range(len(x_t)):
    if y_t[j] < 500:
        ims_to_add.append(x_t[j])
print(ims_to_add)

#%%
#find images with labels that are fewer than 500
#ims_to_add_t=[0,6,16,19,20,21]
import cv2

#both_img = img.copy()
 
# flip img horizontally, vertically,
# and both axes with flip()




new_gauss_X_list=[]
new_gauss_y_list=[]

new_horz_X_list=[]
new_horz_y_list=[]

new_vert_X_list=[]
new_vert_y_list=[]
for i in range(len(X_train)):
#for i in range(20000):
    #print("at i:",i,y_train[i])
    if y_train[i] in ims_to_add:
        #print("match found for i:",i,y_train[i])
        new_gauss_X_list.append(noisy('gauss',X_train[i]))
        new_gauss_y_list.append(y_train[i])
        

        new_horz_X_list.append(cv2.flip( X_train[i], 0 ))
        new_horz_y_list.append(y_train[i])
        new_vert_X_list.append(cv2.flip( X_train[i], 1 ))
        new_vert_y_list.append(y_train[i])

add_X_train=np.array([new_gauss_X_list]).reshape(len(new_gauss_X_list),32,32,3)
add_horz_X_train=np.array([new_horz_X_list]).reshape(len(new_horz_X_list),32,32,3)
add_vert_X_train=np.array([new_vert_X_list]).reshape(len(new_vert_X_list),32,32,3)

add_y_train=np.array([new_gauss_y_list]).reshape(len(new_gauss_y_list),)
add_horz_y_train=np.array([new_horz_y_list]).reshape(len(new_horz_y_list),)
add_vert_y_train=np.array([new_vert_y_list]).reshape(len(new_vert_y_list),)

print("add_X_train:",add_X_train.shape)
print("add_y_train:",add_y_train.shape)
print("add_vert_X_train:",add_vert_X_train.shape)
print("add_vert_y_train:",add_vert_y_train.shape)
plt.figure(figsize=(3,3))
plt.imshow(add_vert_X_train[0])
print("add_horz_X_train:",add_horz_X_train.shape)
print("add_horz_y_train:",add_horz_y_train.shape)
plt.figure(figsize=(3,3))
plt.imshow(add_horz_X_train[0])



#%%
X_new_train=X_train.copy()
y_new_train=y_train.copy()

X_new_train=np.append(X_new_train,add_X_train)
#.reshape(-1,32,32,3)
X_new_train=np.append(X_new_train,add_vert_X_train)
X_new_train=np.append(X_new_train,add_horz_X_train).reshape(-1,32,32,3)

y_new_train=np.append(y_new_train,add_y_train)
y_new_train=np.append(y_new_train,add_vert_y_train)
y_new_train=np.append(y_new_train,add_horz_y_train)

print("New X_train shape",X_new_train.shape)
print("New y_train shape",y_new_train.shape)



#%%
#plot a randomly selected image
index_train_a=random.randint(0, len(add_X_train))
print("Chosen Index in training set is:",index_train_a)
image_train_a = add_X_train[index_train_a]
plt.figure(figsize=(1,1))
plt.imshow(image_train_a)
print("Picture is of type:",add_y_train[index_train_a])

#count of each sign type
#n_classes = len(set(y_train))
plt.figure(2,figsize=(6,5))
count_signtypes_a={}
for stype_a in add_y_train:
    if stype_a not in count_signtypes_a:
        count_signtypes_a[stype_a]=1
    else:
        count_signtypes_a[stype_a] +=1

lists_t_a = sorted(count_signtypes_a.items()) # sorted by key, return a list of tuples

x_t_a, y_t_a = zip(*lists_t_a) # unpack a list of pairs into two tuples

plt.scatter(x_t_a, y_t_a)
#plt.bar(range(len(count_signtypes)),count_signtypes.values())
plt.bar(x_t_a,y_t_a)
plt.show()   
#print(count_signtypes[0],count_signtypes[1],count_signtypes[2],count_signtypes[3],count_signtypes[4],\
#      count_signtypes[5])
max_pics_train_a=np.argmax(y_t_a)
print("Max number of examples are of type:",x_t_a[max_pics_train_a])
#%%
# TODO: Number of training examples
n_train = X_new_train.shape[0]

# TODO: Number of validation examples
n_validation = X_valid.shape[0]

# TODO: Number of testing examples.
n_test = X_test.shape[0]

# TODO: What's the shape of an traffic sign image?
image_shape = (X_new_train.shape[1],X_new_train.shape[1])

# TODO: How many unique classes/labels there are in the dataset.
n_classes = len(set(y_new_train))

print("Number of training examples =", n_train)
print("Number of validation examples =", n_validation)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)

#%%
#plot a randomly selected image
index_train_a=random.randint(0, len(X_new_train))
print("Chosen Index in training set is:",index_train_a)
image_train_a = X_new_train[index_train_a]
plt.figure(figsize=(1,1))
plt.imshow(image_train_a)
print("Picture is of type:",y_new_train[index_train_a])

#count of each sign type
#n_classes = len(set(y_train))
plt.figure(2,figsize=(6,5))
count_signtypes_a={}
for stype_a in y_new_train:
    if stype_a not in count_signtypes_a:
        count_signtypes_a[stype_a]=1
    else:
        count_signtypes_a[stype_a] +=1

lists_t_a = sorted(count_signtypes_a.items()) # sorted by key, return a list of tuples

x_t_a, y_t_a = zip(*lists_t_a) # unpack a list of pairs into two tuples

plt.scatter(x_t_a, y_t_a)
#plt.bar(range(len(count_signtypes)),count_signtypes.values())
plt.bar(x_t_a,y_t_a)
plt.show()   
#print(count_signtypes[0],count_signtypes[1],count_signtypes[2],count_signtypes[3],count_signtypes[4],\
#      count_signtypes[5])
max_pics_train_a=np.argmax(y_t_a)
print("Max number of examples are of type:",x_t_a[max_pics_train_a])

#%%
#normalize train data
print("Shape of training data",X_new_train.shape)
print("Values of train data at [10,0,0,0:3]",X_new_train[10,0,0,0:3])
#Normalize train
X_new_train_norm=(X_new_train-128.0)/128
print("Normalized values of train data at [10,0,0,0:3]",X_new_train_norm[10,0,0,0:3])

#Normalize Validation
print("Values of valid data at [10,0,0,0:3]",X_valid[10,0,0,0:3])
X_valid_norm=(X_valid-128.0)/128
print("Normalized values of validation data at [10,0,0,0:3]",X_valid_norm[10,0,0,0:3])

#plot orig and normalized image
plt.imshow(X_new_train[1])
plt.figure(3)
plt.imshow(X_new_train_norm[1])

#%%

#shuffle training data -
from sklearn.utils import shuffle

#Shuffle original train data
#plt.imshow(X_train[1])
X_train_s, y_train_s = shuffle(X_new_train, y_new_train)
#plt.figure(2)
#plt.imshow(X_train[1])
#plt.figure(3)
#plt.imshow(X_train_s[1])
#normalized train data
#plt.figure(4)
#plt.imshow(X_train[1])
X_new_train_norm, y_new_train_norm = shuffle(X_new_train_norm, y_new_train)
#plt.figure(5)
#plt.imshow(X_train[1])
#plt.figure(6)
#plt.imshow(X_train_norm[1])

#Shuffle original valid data
#plt.imshow(X_valid[1])
X_valid_s, y_valid_s = shuffle(X_valid, y_valid)
#normalized validation data -- not shuffled
#X_valid_norm, y_valid_norm = (X_valid_norm, y_valid)
#plt.figure(2)
#plt.imshow(X_valid[1])
#plt.figure(3)
#plt.imshow(X_valid_s[1])

#%%
#import tensor flow
#set epochs and batch size
import tensorflow as tf

EPOCHS = 10
BATCH_SIZE = 128

#%%
from tensorflow.contrib.layers import flatten

def LeNet(x):    
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1
    
    # SOLUTION: Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 3, 6), mean = mu, stddev = sigma))
    conv1_b = tf.Variable(tf.zeros(6))
    conv1   = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b

    # SOLUTION: Activation.
    conv1 = tf.nn.relu(conv1)

    # SOLUTION: Pooling. Input = 28x28x6. Output = 14x14x6.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # SOLUTION: Layer 2: Convolutional. Output = 10x10x16.
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean = mu, stddev = sigma))
    conv2_b = tf.Variable(tf.zeros(16))
    conv2   = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b
    
    # SOLUTION: Activation.
    conv2 = tf.nn.relu(conv2)

    # SOLUTION: Pooling. Input = 10x10x16. Output = 5x5x16.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # SOLUTION: Flatten. Input = 5x5x16. Output = 400.
    fc0   = flatten(conv2)
    
    # SOLUTION: Layer 3: Fully Connected. Input = 400. Output = 120.
    fc1_W = tf.Variable(tf.truncated_normal(shape=(400, 120), mean = mu, stddev = sigma))
    fc1_b = tf.Variable(tf.zeros(120))
    fc1   = tf.matmul(fc0, fc1_W) + fc1_b
    
    # SOLUTION: Activation.
    fc1    = tf.nn.relu(fc1)

    # SOLUTION: Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2_W  = tf.Variable(tf.truncated_normal(shape=(120, 84), mean = mu, stddev = sigma))
    fc2_b  = tf.Variable(tf.zeros(84))
    fc2    = tf.matmul(fc1, fc2_W) + fc2_b
    
    # SOLUTION: Activation.
    fc2    = tf.nn.relu(fc2)

    # SOLUTION: Layer 5: Fully Connected. Input = 84. Output = 10.
    fc3_W  = tf.Variable(tf.truncated_normal(shape=(84, 43), mean = mu, stddev = sigma))
    fc3_b  = tf.Variable(tf.zeros(43))
    logits = tf.matmul(fc2, fc3_W) + fc3_b
    
    return logits
#%%

#Features and labels
rgb_channel=3
#rgb_channel=1
x = tf.placeholder(tf.float32, (None, 32, 32, rgb_channel))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, n_classes)
keep_prob=tf.placeholder(tf.float32)
#print(n_classes)

#%%
#Training pipeline
rate = 0.001

logits = LeNet(x)
#logits = LeNet_gray(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)

##L2 reg
#beta=0.01
#regularizers = tf.nn.l2_loss(weights['wc1']) + tf.nn.l2_loss(weights['wc2']) + tf.nn.l2_loss(weights['wd1']) + tf.nn.l2_loss(weights['wd2']) \
#    + tf.nn.l2_loss(weights['out'])
#loss_operation = tf.reduce_mean(loss_operation + beta * regularizers)


##ADAM optimizer
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)

#Gradient descent optimizer

#optimizer = tf.train.GradientDescentOptimizer(learning_rate=rate)
#training_operation = optimizer.minimize(loss_operation)

#%%
drop_prob_valid=1.0

#Model Eval
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y,keep_prob:drop_prob_valid})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

#%%
#set flag to use normalized or unnormalized data
use_normalized_train=False
#use_normalized_train=True
use_normalized_validation=False
#use_normalized_validation=True

use_drop_out=False
#use_drop_out=True

if use_drop_out:
    drop_prob_train=0.8   
else:
    drop_prob_train=1.0
    
print("Normalized Train:{}, Normalized Valid:{}, Use dropout:{}".format(use_normalized_train,use_normalized_validation,use_drop_out))


#Train the model
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)
    
    print("Training...")
    #print()
    for i in range(EPOCHS):
        #already shuffled
        #X_train, y_train = shuffle(X_train, y_train)
        if use_normalized_train:
            X_new_train_norm, y_new_train_norm = shuffle(X_new_train_norm, y_new_train_norm)
        else:
            X_train_s, y_train_s = shuffle(X_train_s, y_train_s)
               
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            #batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            if use_normalized_train:
                batch_x, batch_y = X_new_train_norm[offset:end], y_new_train_norm[offset:end]
                sess.run(training_operation, feed_dict={x: batch_x, y: batch_y,keep_prob:drop_prob_train})
            else:
                batch_x, batch_y = X_train_s[offset:end], y_train_s[offset:end]
                sess.run(training_operation, feed_dict={x: batch_x, y: batch_y,keep_prob:drop_prob_train})
                
        
        if use_normalized_validation:
            validation_accuracy = evaluate(X_valid_norm, y_valid)
        else:
            validation_accuracy = evaluate(X_valid, y_valid)
        
        
        print("EPOCH {} ...".format(i+1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()
        
    saver.save(sess, './lenet_german_traffic_sign')
    print("Model saved")