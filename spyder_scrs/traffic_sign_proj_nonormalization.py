# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 19:23:15 2017

@author: atpandey
"""

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
### Data exploration visualization code goes here.
### Feel free to use as many code cells as needed.

#Plot samples and labels from training set

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
#Plot samples and labels from validation set
import random
import matplotlib.pyplot as plt
# Visualizations will be shown in the notebook.
#%matplotlib inline

#plot a randomly selected image
index_valid=random.randint(0, len(X_valid))
print("Chosen Index in Validation set is:",index_valid)
image_valid = X_valid[index_valid]
plt.figure(figsize=(1,1))
plt.imshow(image_valid)
print("Picture is of type:",y_valid[index_valid])

#count of each sign type
#n_classes = len(set(y_train))
plt.figure(2,figsize=(6,5))
count_signtypes_v={}
for stype in y_valid:
    if stype not in count_signtypes_v:
        count_signtypes_v[stype]=1
    else:
        count_signtypes_v[stype] +=1

lists_v = sorted(count_signtypes_v.items()) # sorted by key, return a list of tuples

x_v, y_v = zip(*lists_v) # unpack a list of pairs into two tuples

plt.scatter(x_v, y_v)
#plt.bar(range(len(count_signtypes)),count_signtypes.values())
plt.bar(x_v,y_v)
plt.show()   
#print(count_signtypes[0],count_signtypes[1],count_signtypes[2],count_signtypes[3],count_signtypes[4],\
#      count_signtypes[5])
max_pics_train_v=np.argmax(y_v)
print("Max number of examples are of type:",x_v[max_pics_train_v])

#%%
#normalize train data
print("Shape of training data",X_train.shape)
print("Values of train data at [10,0,0,0:3]",X_train[10,0,0,0:3])
#Normalize train
X_train_norm=(X_train-128.0)/128
print("Normalized values of train data at [10,0,0,0:3]",X_train_norm[10,0,0,0:3])

#Normalize Validation
print("Values of valid data at [10,0,0,0:3]",X_valid[10,0,0,0:3])
X_valid_norm=(X_valid-128.0)/128
print("Normalized values of validation data at [10,0,0,0:3]",X_valid_norm[10,0,0,0:3])

#plot orig and normalized image
plt.imshow(X_train[1])
plt.figure(3)
plt.imshow(X_train_norm[1])

#%%

#shuffle training data -
from sklearn.utils import shuffle

#Shuffle original train data
#plt.imshow(X_train[1])
X_train_s, y_train_s = shuffle(X_train, y_train)
#plt.figure(2)
#plt.imshow(X_train[1])
#plt.figure(3)
#plt.imshow(X_train_s[1])
#normalized train data
#plt.figure(4)
#plt.imshow(X_train[1])
X_train_norm, y_train_norm = shuffle(X_train_norm, y_train)
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


def conv2d(x, W, b, strides=1,padarg='SAME'):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding=padarg)
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool(x, k=2,stride=2,padarg='SAME'):
    return tf.nn.max_pool(
        x,
        ksize=[1, k, k, 1],
        strides=[1,stride,stride, 1],
        padding=padarg)
#already set in second cell (len(set(y_train)))
#n_classes=(len(set(y_train)))
keep_prob=tf.placeholder(tf.float32)



###########################################
mu = 0
sigma = 0.1
#Data needs regularization? X-mu/sigma ??
#the random samples for weights needs to be mean mu and std of sigma
weights = {
'wc1': tf.Variable(tf.truncated_normal([5, 5, 3, 6], mean = mu, stddev = sigma)),
'wc2': tf.Variable(tf.truncated_normal([5, 5, 6, 16],mean = mu, stddev = sigma)),
'wd1': tf.Variable(tf.truncated_normal([5*5*16, 120],mean = mu, stddev = sigma)),
'wd2': tf.Variable(tf.truncated_normal([120, 84],mean = mu, stddev = sigma)),
'out': tf.Variable(tf.truncated_normal([84, n_classes],mean = mu, stddev = sigma))}

biases = {
'bc1': tf.Variable(tf.truncated_normal([6], mean = mu, stddev = sigma)),
'bc2': tf.Variable(tf.truncated_normal([16], mean = mu, stddev = sigma)),
'bd1': tf.Variable(tf.truncated_normal([120], mean = mu, stddev = sigma)),
'bd2': tf.Variable(tf.truncated_normal([84], mean = mu, stddev = sigma)),
'out': tf.Variable(tf.truncated_normal([n_classes], mean = mu, stddev = sigma))}



############################################

def LeNet(x):  
    
    #print("shape of X is:",x.shape)
    #print("lenetX",x[0,19:21,14:20,0])
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer

    
    
    # Layer 1: Convolutional. Input = 32x32x3. Output = 28x28x6.
    
    conv1=conv2d(x, weights['wc1'], biases['bc1'], 1,'VALID')
    
    #Activation1.

    
    
    #Pooling. Input = 28x28x6. Output = 14x14x6.
    conv1=maxpool(conv1, 2,2,'VALID')
    
    #Dropout
    conv1 = tf.nn.dropout(conv1, keep_prob)

    
    
    # TODO: Layer 2: Convolutional. Output = 10x10x16.
    conv2=conv2d(conv1, weights['wc2'], biases['bc2'], 1,'VALID')

    
    
    # TODO: Activation.
   
    
    # TODO: Pooling. Input = 10x10x16. Output = 5x5x16.
    conv2=maxpool(conv2, 2,2,'VALID')
    #Dropout
    conv2 = tf.nn.dropout(conv2, keep_prob)

    
    
    # TODO: Flatten. Input = 5x5x16. Output = 400.
    #print("WW1",weights['wd1'].get_shape().as_list()[0])
    #print("WW2",weights['wd1'].get_shape().as_list())
    
    #fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1=flatten(conv2)
    #print("WW3",fc1.get_shape().as_list())
    
    # TODO: Layer 3: Fully Connected. Input = 400. Output = 120.
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    
    
    # TODO: Activation.
    fc1 = tf.nn.relu(fc1)
    #drop out
    fc1 = tf.nn.dropout(fc1, keep_prob)
    
    
    # TODO: Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2 = tf.add(tf.matmul(fc1, weights['wd2']), biases['bd2'])
    # TODO: Activation.
    fc2=tf.nn.relu(fc2)
    # TODO: Layer 5: Fully Connected. Input = 84. Output = 43.
    logits = tf.add(tf.matmul(fc2, weights['out']), biases['out'])
    #print('WW4',logits.get_shape().as_list())
    return logits

#%%

#Features and labels
rgb_channel=3
#rgb_channel=1
x = tf.placeholder(tf.float32, (None, 32, 32, rgb_channel))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, n_classes)
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
#use_normalized_train=False
use_normalized_train=True
#use_normalized_validation=False
use_normalized_validation=True

use_drop_out=False
#use_drop_out=True

if use_drop_out:
    #drop_prob_train=0.8
    drop_prob_train=0.5
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
            X_train_norm, y_train_norm = shuffle(X_train_norm, y_train_norm)
        else:
            X_train_s, y_train_s = shuffle(X_train_s, y_train_s)
               
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            #batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            if use_normalized_train:
                batch_x, batch_y = X_train_norm[offset:end], y_train_norm[offset:end]
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
#%%