# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 19:46:38 2017

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
#n_classes=len(np.unique(y_train))

print("Number of training examples =", n_train)
print("Number of validation examples =", n_validation)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)

#%%
#### Data exploration visualization code goes here.
#### Feel free to use as many code cells as needed.
#
##Plot samples and labels from training set
#
#import random
#import matplotlib.pyplot as plt
## Visualizations will be shown in the notebook.
##%matplotlib inline
#
##plot a randomly selected image
#index_train=random.randint(0, len(X_train))
#print("Chosen Index in training set is:",index_train)
#image_train = X_train[index_train]
#plt.figure(figsize=(1,1))
#plt.imshow(image_train)
#print("Picture is of type:",y_train[index_train])
#
##count of each sign type
##n_classes = len(set(y_train))
#plt.figure(2,figsize=(6,5))
#count_signtypes={}
#for stype in y_train:
#    if stype not in count_signtypes:
#        count_signtypes[stype]=1
#    else:
#        count_signtypes[stype] +=1
#
#lists_t = sorted(count_signtypes.items()) # sorted by key, return a list of tuples
#
#x_t, y_t = zip(*lists_t) # unpack a list of pairs into two tuples
#
#plt.scatter(x_t, y_t)
##plt.bar(range(len(count_signtypes)),count_signtypes.values())
#plt.bar(x_t,y_t)
#plt.show()   
##print(count_signtypes[0],count_signtypes[1],count_signtypes[2],count_signtypes[3],count_signtypes[4],\
##      count_signtypes[5])
#max_pics_train=np.argmax(y_t)
#print("Max number of examples are of type:",x_t[max_pics_train])

#%%
### Data exploration visualization goes here.
### Feel free to use as many code cells as needed.
import matplotlib.pyplot as plt
import random
# Visualizations will be shown in the notebook.
#%matplotlib inline

# show image of 10 random data points
fig, axs = plt.subplots(2,5, figsize=(15, 6))
fig.subplots_adjust(hspace = .2, wspace=.001)
axs = axs.ravel()
for i in range(10):
    index = random.randint(0, len(X_train))
    image = X_train[index]
    axs[i].axis('off')
    axs[i].imshow(image)
    axs[i].set_title(y_train[index])

# histogram of label frequency
plt.figure(2)
hist, bins = np.histogram(y_train, bins=n_classes)
width = 0.7 * (bins[1] - bins[0])
center = (bins[:-1] + bins[1:]) / 2
plt.bar(center, hist, align='center', width=width)
plt.show()

#%%
### Preprocess the data here.
### Feel free to use as many code cells as needed.

# Convert to grayscale
X_train_rgb = X_train
X_train_gry = np.sum(X_train/3, axis=3, keepdims=True)

X_valid_rgb = X_valid
X_valid_gry = np.sum(X_valid/3, axis=3, keepdims=True)

X_test_rgb = X_test
X_test_gry = np.sum(X_test/3, axis=3, keepdims=True)

print('RGB shape:', X_train_rgb.shape)
print('Grayscale shape:', X_train_gry.shape)

#%%
X_train = X_train_gry
X_valid = X_valid_gry
X_test = X_test_gry

print('done')

#%%
# Visualize rgb vs grayscale
n_rows = 8
n_cols = 10
offset = 9000
fig, axs = plt.subplots(n_rows,n_cols, figsize=(18, 14))
fig.subplots_adjust(hspace = .1, wspace=.001)
axs = axs.ravel()
for j in range(0,n_rows,2):
    for i in range(n_cols):
        index = i + j*n_cols
        image = X_train_rgb[index + offset]
        axs[index].axis('off')
        axs[index].imshow(image)
    for i in range(n_cols):
        index = i + j*n_cols + n_cols 
        image = X_train_gry[index + offset - n_cols].squeeze()
        axs[index].axis('off')
        axs[index].imshow(image, cmap='gray')

#%%
print("train labels order")
print(y_train[0:500])
print("valid labels order")
print(y_valid[0:500])

#%%
print("mean train:",np.mean(X_train))
print("mean valid:",np.mean(X_valid))
print("mean test:",np.mean(X_test))

#%%
## Normalize the train and test datasets to (-1,1)

X_train_normalized = (X_train - 128.)/128 
X_valid_normalized = (X_valid - 128.)/128                     
X_test_normalized = (X_test - 128.)/128

print("mean train normalized :",np.mean(X_train_normalized))
print("mean valid normalized :",np.mean(X_valid_normalized))
print("mean test normalized :",np.mean(X_test_normalized))

#%%
print("Original shape:", X_train.shape)
print("Normalized shape:", X_train_normalized.shape)
fig, axs = plt.subplots(1,2, figsize=(10, 3))
axs = axs.ravel()

axs[0].axis('off')
axs[0].set_title('normalized')
axs[0].imshow(X_train_normalized[1000].squeeze(), cmap='gray')

axs[1].axis('off')
axs[1].set_title('original')
axs[1].imshow(X_train[1000].squeeze(), cmap='gray')

#%%
print("bin count for each sample",np.bincount(y_train))
print("minimum samples for any label:", min(np.bincount(y_train)))

#%%
## Shuffle the training dataset

from sklearn.utils import shuffle

X_train, y_train = shuffle(X_train_normalized, y_train)
X_valid, y_valid = shuffle(X_valid_normalized, y_valid)

print('Shuffle of train and validation data done')

#%%
import tensorflow as tf

EPOCHS = 10
#EPOCHS = 20
#EPOCHS = 30
BATCH_SIZE = 128

print("Epoch and batch size setting done")

#%%
from tensorflow.contrib.layers import flatten

def LeNet(x):    
    # Hyperparameters
    mu = 0
    sigma = 0.1
    
    # TODO: Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    W1 = tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 6), mean = mu, stddev = sigma))
    x = tf.nn.conv2d(x, W1, strides=[1, 1, 1, 1], padding='VALID')
    b1 = tf.Variable(tf.zeros(6))
    x = tf.nn.bias_add(x, b1)
    print("layer 1 shape:",x.get_shape())

    # TODO: Activation.
    x = tf.nn.relu(x)
    
    # TODO: Pooling. Input = 28x28x6. Output = 14x14x6.
    x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    
    # TODO: Layer 2: Convolutional. Output = 10x10x16.
    W2 = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean = mu, stddev = sigma))
    x = tf.nn.conv2d(x, W2, strides=[1, 1, 1, 1], padding='VALID')
    b2 = tf.Variable(tf.zeros(16))
    x = tf.nn.bias_add(x, b2)
                     
    # TODO: Activation.
    x = tf.nn.relu(x)

    # TODO: Pooling. Input = 10x10x16. Output = 5x5x16.
    x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # TODO: Flatten. Input = 5x5x16. Output = 400.
    x = flatten(x)
    
    # TODO: Layer 3: Fully Connected. Input = 400. Output = 120.
    W3 = tf.Variable(tf.truncated_normal(shape=(400, 120), mean = mu, stddev = sigma))
    b3 = tf.Variable(tf.zeros(120))    
    x = tf.add(tf.matmul(x, W3), b3)
    
    # TODO: Activation.
    x = tf.nn.relu(x)
    
    # Dropout
    x = tf.nn.dropout(x, keep_prob)

    # TODO: Layer 4: Fully Connected. Input = 120. Output = 84.
    W4 = tf.Variable(tf.truncated_normal(shape=(120, 84), mean = mu, stddev = sigma))
    b4 = tf.Variable(tf.zeros(84)) 
    x = tf.add(tf.matmul(x, W4), b4)
    
    # TODO: Activation.
    x = tf.nn.relu(x)
    
    # Dropout
    x = tf.nn.dropout(x, keep_prob)

    # TODO: Layer 5: Fully Connected. Input = 84. Output = 43.
    W5 = tf.Variable(tf.truncated_normal(shape=(84, 43), mean = mu, stddev = sigma))
    b5 = tf.Variable(tf.zeros(43)) 
    logits = tf.add(tf.matmul(x, W5), b5)
    
    return logits

print('Lenet done')

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

#n_classes=


# tf Graph input
#x = tf.placeholder(tf.float32, [None, 32, 32, 1])
#y = tf.placeholder(tf.float32, [None, n_classes])
#keep_prob = tf.placeholder(tf.float32)


def LeNet_gray(x):  
    
    #print("shape of X is:",x.shape)
    #print("lenetX",x[0,19:21,14:20,0])
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1
    #Data needs regularization? X-mu/sigma ??
    #the random samples for weights needs to be mean mu and std of sigma
    #################################################################################
    
    weights = {
    'wc1': tf.Variable(tf.truncated_normal([5, 5, 1, 6], mean = mu, stddev = sigma)),
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

    ###########################################################################################

    
    
    # TODO: Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    
    conv1=conv2d(x, weights['wc1'], biases['bc1'], 1,'VALID')
    #conv1=tf.nn.conv2d(x,weights['wc1'], strides=[1, 1, 1, 1], padding='VALID')
    #conv1 = tf.nn.bias_add(conv1, biases['bc1'])
    
    # TODO: Activation.
    #conv1 = tf.nn.relu(conv1)
    
    
    # TODO: Pooling. Input = 28x28x6. Output = 14x14x6.
    conv1=maxpool(conv1, 2,2,'VALID')
    #conv1=tf.nn.max_pool(
    #    conv1,
    #    ksize=[1, 2, 2, 1],
    #    strides=[1,2,2, 1],
    #    padding='SAME')
    
    
    # TODO: Layer 2: Convolutional. Output = 10x10x16.
    conv2=conv2d(conv1, weights['wc2'], biases['bc2'], 1,'VALID')
    #conv2=tf.nn.conv2d(conv1,weights['wc2'], strides=[1, 1, 1, 1], padding='VALID')
    #conv2 = tf.nn.bias_add(conv2, biases['bc2'])
    
    
    # TODO: Activation.
    #conv2 = tf.nn.relu(conv2)
    
    # TODO: Pooling. Input = 10x10x16. Output = 5x5x16.
    conv2=maxpool(conv2, 2,2,'VALID')
    #conv2=tf.nn.max_pool(
    #    conv2,
    #    ksize=[1, 2, 2, 1],
    #    strides=[1,2,2, 1],
    #    padding='SAME')
    #print("WW0",conv2.get_shape().as_list())
    
    
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
    fc1 = tf.nn.dropout(fc1, keep_prob)
    # TODO: Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2 = tf.add(tf.matmul(fc1, weights['wd2']), biases['bd2'])
    # TODO: Activation.
    fc2=tf.nn.relu(fc2)
    fc2 = tf.nn.dropout(fc2, keep_prob)
    # TODO: Layer 5: Fully Connected. Input = 84. Output = 43.
    logits = tf.add(tf.matmul(fc2, weights['out']), biases['out'])
    #print('WW4',logits.get_shape().as_list())
    return logits

#%%
tf.reset_default_graph() 

x = tf.placeholder(tf.float32, (None, 32, 32, 1))
y = tf.placeholder(tf.int32, (None))
keep_prob = tf.placeholder(tf.float32) # probability to keep units
one_hot_y = tf.one_hot(y, 43)

print('Placeholders done')

#%%
### Train your model here.
### Feel free to use as many code cells as needed.

#rate = 0.0009
rate = 0.001
#rate =0.0005
#logits = LeNet(x)
logits = LeNet_gray(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)

print("Train ops setup done")
#%%
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

print('Prediction validation setup done')

#%%
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)
    
    print("Training...")
    print()
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 0.5})
            
        validation_accuracy = evaluate(X_valid, y_valid)
        print("EPOCH {} ...".format(i+1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()
        
    saver.save(sess, './lenet_gray')
    print("Model saved")


#%%
#Run1: greyscale,train/valid/test normalized, epoch=10, batch=128, learnrate=0.001,dropout
#0.930 accuracy
#Run2: greyscale,train/valid/test normalized, epoch=20, batch=128, learnrate=0.001,dropout
#0.957, from 10 -20 epoch accuracy fluctuates between .93 to .95
#Run3: greyscale,train/valid/test normalized, epoch=20, batch=128, learnrate=0.0005,dropout
# at epoch 10 --> 0.877, 0.928, accuracy mostly increases progressively
#Run4: greyscale,train/valid/test normalized, epoch=30, batch=128, learnrate=0.0005,dropout
# at epoch 10 --> 0.908, at 20 0.946, at 30 0.955
#Run5:  Lenet_gray greyscale,train/valid/test normalized, epoch=30, batch=128, learnrate=0.0005 bias without mu and sigma !!
# at 30, 0.876 --> no dropout
#Run6 :  Lenet_gray greyscale,train/valid/test normalized, epoch=30, batch=128, learnrate=0.0005, bias random_normal with mu and sigma
# at 30, 0.896 --> no dropout
#Run7 :  Lenet_gray greyscale,train/valid/test normalized, epoch=30, batch=128, learnrate=0.0005, weight and bias truncated_normal with mu and sigma
# at 30,0.899 -- accuracy fluctuates around epoch 10 between 0.883 - 0.890 --> no dropout
#Run8 :  Lenet_gray greyscale,train/valid/test normalized, epoch=30, batch=128, learnrate=0.0005, weight truncated_normal, bias zeros
# at 30, 0.901  --> no dropout
#Run9  :  Lenet_gray greyscale,train/valid/test normalized, epoch=30, batch=128, learnrate=0.0005, droput at layer 3 and 4 weight truncated_normal, bias zeros
# at 10,.889 at 30, 0.945
#Run10: Lenet_gray greyscale,train/valid/test normalized, epoch=10, batch=128, learnrate=0.001, droput at layer 3 and 4 weight truncated_normal, bias zeros
# at 10, 0.924

#%%
########################################################
##%%
##Plot samples and labels from validation set
#import random
#import matplotlib.pyplot as plt
## Visualizations will be shown in the notebook.
##%matplotlib inline
#
##plot a randomly selected image
#index_valid=random.randint(0, len(X_valid))
#print("Chosen Index in Validation set is:",index_valid)
#image_valid = X_valid[index_valid]
#plt.figure(figsize=(1,1))
#plt.imshow(image_valid)
#print("Picture is of type:",y_valid[index_valid])
#
##count of each sign type
##n_classes = len(set(y_train))
#plt.figure(2,figsize=(6,5))
#count_signtypes_v={}
#for stype in y_valid:
#    if stype not in count_signtypes_v:
#        count_signtypes_v[stype]=1
#    else:
#        count_signtypes_v[stype] +=1
#
#lists_v = sorted(count_signtypes_v.items()) # sorted by key, return a list of tuples
#
#x_v, y_v = zip(*lists_v) # unpack a list of pairs into two tuples
#
#plt.scatter(x_v, y_v)
##plt.bar(range(len(count_signtypes)),count_signtypes.values())
#plt.bar(x_v,y_v)
#plt.show()   
##print(count_signtypes[0],count_signtypes[1],count_signtypes[2],count_signtypes[3],count_signtypes[4],\
##      count_signtypes[5])
#max_pics_train_v=np.argmax(y_v)
#print("Max number of examples are of type:",x_v[max_pics_train_v])
#
##%%
##normalize train data
#print("Shape of training data",X_train.shape)
#print("Values of train data at [10,0,0,0:3]",X_train[10,0,0,0:3])
##Normalize train
#X_train_norm=(X_train-128.0)/128
#print("Normalized values of train data at [10,0,0,0:3]",X_train_norm[10,0,0,0:3])
#
##Normalize Validation
#print("Values of valid data at [10,0,0,0:3]",X_valid[10,0,0,0:3])
#X_valid_norm=(X_valid-128.0)/128
#print("Normalized values of validation data at [10,0,0,0:3]",X_valid_norm[10,0,0,0:3])
#
##plot orig and normalized image
#plt.imshow(X_train[1])
#plt.figure(3)
#plt.imshow(X_train_norm[1])
#
##%%
##shuffle training data -
#from sklearn.utils import shuffle
#
##Shuffle original train data
##plt.imshow(X_train[1])
#X_train_s, y_train_s = shuffle(X_train, y_train)
##plt.figure(2)
##plt.imshow(X_train[1])
##plt.figure(3)
##plt.imshow(X_train_s[1])
##normalized train data
##plt.figure(4)
##plt.imshow(X_train[1])
#X_train_norm, y_train_norm = shuffle(X_train_norm, y_train)
##plt.figure(5)
##plt.imshow(X_train[1])
##plt.figure(6)
##plt.imshow(X_train_norm[1])
#
##Shuffle original valid data
##plt.imshow(X_valid[1])
#X_valid_s, y_valid_s = shuffle(X_valid, y_valid)
##normalized validation data -- not shuffled
##X_valid_norm, y_valid_norm = (X_valid_norm, y_valid)
##plt.figure(2)
##plt.imshow(X_valid[1])
##plt.figure(3)
##plt.imshow(X_valid_s[1])
#
##%%
##import tensor flow
##set epochs and batch size
#import tensorflow as tf
#
#EPOCHS = 10
#BATCH_SIZE = 128
##%%
#from tensorflow.contrib.layers import flatten
#
#
#def conv2d(x, W, b, strides=1,padarg='SAME'):
#    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding=padarg)
#    x = tf.nn.bias_add(x, b)
#    return tf.nn.relu(x)
#
#
#def maxpool(x, k=2,stride=2,padarg='SAME'):
#    return tf.nn.max_pool(
#        x,
#        ksize=[1, k, k, 1],
#        strides=[1,stride,stride, 1],
#        padding=padarg)
#
##n_classes=
#
#
## tf Graph input
##x = tf.placeholder(tf.float32, [None, 32, 32, 1])
##y = tf.placeholder(tf.float32, [None, n_classes])
##keep_prob = tf.placeholder(tf.float32)
#
#
#def LeNet_gray(x):  
#    
#    #print("shape of X is:",x.shape)
#    #print("lenetX",x[0,19:21,14:20,0])
#    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
#    mu = 0
#    sigma = 0.1
#    #Data needs regularization? X-mu/sigma ??
#    #the random samples for weights needs to be mean mu and std of sigma
#    #################################################################################
#    
#    weights = {
#    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 6], mean = mu, stddev = sigma)),
#    'wc2': tf.Variable(tf.random_normal([5, 5, 6, 16],mean = mu, stddev = sigma)),
#    'wd1': tf.Variable(tf.random_normal([5*5*16, 120],mean = mu, stddev = sigma)),
#     'wd2': tf.Variable(tf.random_normal([120, 84],mean = mu, stddev = sigma)),
#    'out': tf.Variable(tf.random_normal([84, n_classes],mean = mu, stddev = sigma))}
#
#    biases = {
#    'bc1': tf.Variable(tf.random_normal([6])),
#    'bc2': tf.Variable(tf.random_normal([16])),
#    'bd1': tf.Variable(tf.random_normal([120])),
#    'bd2': tf.Variable(tf.random_normal([84])),
#    'out': tf.Variable(tf.random_normal([n_classes]))}
#
#    ###########################################################################################
#
#    
#    
#    # TODO: Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
#    
#    conv1=conv2d(x, weights['wc1'], biases['bc1'], 1,'VALID')
#    #conv1=tf.nn.conv2d(x,weights['wc1'], strides=[1, 1, 1, 1], padding='VALID')
#    #conv1 = tf.nn.bias_add(conv1, biases['bc1'])
#    
#    # TODO: Activation.
#    #conv1 = tf.nn.relu(conv1)
#    
#    
#    # TODO: Pooling. Input = 28x28x6. Output = 14x14x6.
#    conv1=maxpool(conv1, 2,2,'SAME')
#    #conv1=tf.nn.max_pool(
#    #    conv1,
#    #    ksize=[1, 2, 2, 1],
#    #    strides=[1,2,2, 1],
#    #    padding='SAME')
#    
#    
#    # TODO: Layer 2: Convolutional. Output = 10x10x16.
#    conv2=conv2d(conv1, weights['wc2'], biases['bc2'], 1,'VALID')
#    #conv2=tf.nn.conv2d(conv1,weights['wc2'], strides=[1, 1, 1, 1], padding='VALID')
#    #conv2 = tf.nn.bias_add(conv2, biases['bc2'])
#    
#    
#    # TODO: Activation.
#    #conv2 = tf.nn.relu(conv2)
#    
#    # TODO: Pooling. Input = 10x10x16. Output = 5x5x16.
#    conv2=maxpool(conv2, 2,2,'SAME')
#    #conv2=tf.nn.max_pool(
#    #    conv2,
#    #    ksize=[1, 2, 2, 1],
#    #    strides=[1,2,2, 1],
#    #    padding='SAME')
#    #print("WW0",conv2.get_shape().as_list())
#    
#    
#    # TODO: Flatten. Input = 5x5x16. Output = 400.
#    #print("WW1",weights['wd1'].get_shape().as_list()[0])
#    #print("WW2",weights['wd1'].get_shape().as_list())
#    
#    #fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
#    fc1=flatten(conv2)
#    #print("WW3",fc1.get_shape().as_list())
#    
#    # TODO: Layer 3: Fully Connected. Input = 400. Output = 120.
#    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
#    
#    
#    # TODO: Activation.
#    fc1 = tf.nn.relu(fc1)
#    # TODO: Layer 4: Fully Connected. Input = 120. Output = 84.
#    fc2 = tf.add(tf.matmul(fc1, weights['wd2']), biases['bd2'])
#    # TODO: Activation.
#    fc2=tf.nn.relu(fc2)
#    # TODO: Layer 5: Fully Connected. Input = 84. Output = 43.
#    logits = tf.add(tf.matmul(fc2, weights['out']), biases['out'])
#    #print('WW4',logits.get_shape().as_list())
#    return logits
#
##%%
##Features and labels
##rgb_channel=3
#rgb_channel=1
#x = tf.placeholder(tf.float32, (None, 32, 32, rgb_channel))
#y = tf.placeholder(tf.int32, (None))
#one_hot_y = tf.one_hot(y, n_classes)
##print(n_classes)
#
##%%
##Training pipeline
#rate = 0.001
#
##logits = LeNet(x)
#logits = LeNet_gray(x)
#cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
#loss_operation = tf.reduce_mean(cross_entropy)
#
###L2 reg
##beta=0.01
##regularizers = tf.nn.l2_loss(weights['wc1']) + tf.nn.l2_loss(weights['wc2']) + tf.nn.l2_loss(weights['wd1']) + tf.nn.l2_loss(weights['wd2']) \
##    + tf.nn.l2_loss(weights['out'])
##loss_operation = tf.reduce_mean(loss_operation + beta * regularizers)
#
#
###ADAM optimizer
#optimizer = tf.train.AdamOptimizer(learning_rate = rate)
#training_operation = optimizer.minimize(loss_operation)
#
##Gradient descent optimizer
#
##optimizer = tf.train.GradientDescentOptimizer(learning_rate=rate)
##training_operation = optimizer.minimize(loss_operation)
##%%
#drop_prob_valid=1.0
#
##Model Eval
#correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
#accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#saver = tf.train.Saver()
#
#def evaluate(X_data, y_data):
#    num_examples = len(X_data)
#    total_accuracy = 0
#    sess = tf.get_default_session()
#    for offset in range(0, num_examples, BATCH_SIZE):
#        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
#        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y,keep_prob:drop_prob_valid})
#        total_accuracy += (accuracy * len(batch_x))
#    return total_accuracy / num_examples
##%%
#
#
##%%
###Add gray scale the pictures in trg and validation set
#
##Grayscaled pictures
##convert rgb pictures to gray scale
#def rgb2gray(x):
#    return np.dot(x[...,:3], [0.299, 0.587, 0.114])
#X_train_gray_s=rgb2gray(X_train_s)
#X_valid_gray=rgb2gray(X_valid)
#
#print("Shape of X_train_gray is:",X_train_gray_s.shape)
#print("Shape of X_valid_gray is:",X_valid_gray.shape)
#X_train_gray_s=X_train_gray_s.reshape(-1,32,32,1)
#print("Shape of X_train_gray is:",X_train_gray_s.shape)
#X_valid_gray=X_valid_gray.reshape(-1,32,32,1)
#print("Shape of X_valid_gray is:",X_valid_gray.shape)
#
#
#X_train_gray_norm=(X_train_gray_s-128.0)/128
#print("Shape of X_train_gray_norm is:",X_train_gray_norm.shape)
#X_valid_gray_norm=(X_valid_gray-128.0)/128
#print("Shape of X_valid_gray_norm is:",X_valid_gray_norm.shape)
#
#index_t = random.randint(0, len(X_train_s))
#print("Index Train:",index_t)
#image_s = X_train_s[index_t].squeeze()
#plt.figure(figsize=(4,4))
#plt.imshow(image_s, cmap="gray")
##print(y_train_s[index_t])
#plt.title(y_train_s[index_t])
#
#image_g = X_train_gray_s[index_t].squeeze()
#plt.figure(2,figsize=(4,4))
#plt.imshow(image_g, cmap="gray")
#plt.title(y_train_s[index_t])
#
#image_gn = X_train_gray_norm[index_t].squeeze()
#plt.figure(3,figsize=(4,4))
#plt.imshow(image_gn, cmap="gray")
#plt.title(y_train_s[index_t])
#
#index_v = random.randint(0, len(X_valid))
#print("Index Valid:",index_v)
#image_v = X_valid[index_v].squeeze()
#plt.figure(figsize=(4,4))
#plt.imshow(image_v, cmap="gray")
#plt.title(y_valid[index_v])
#
#image_vg = X_valid_gray[index_v].squeeze()
#plt.figure(figsize=(4,4))
#plt.imshow(image_vg, cmap="gray")
#plt.title(y_valid[index_v])
#
#image_vgn = X_valid_gray_norm[index_v].squeeze()
#plt.figure(figsize=(4,4))
#plt.imshow(image_vgn, cmap="gray")
#plt.title(y_valid[index_v])
#
##%%
##set flag to use normalized or unnormalized data
##use_normalized_train=False
#use_normalized_train=True
##use_normalized_validation=False
#use_normalized_validation=True
#
#use_drop_out=False
##use_drop_out=True
#
#if use_drop_out:
#    drop_prob_train=0.8   
#else:
#    drop_prob_train=1.0
#    
#print("Normalized Train:{}, Normalized Valid:{}, Use dropout:{}".format(use_normalized_train,use_normalized_validation,use_drop_out))
#
#
##Train the model
#with tf.Session() as sess:
#    sess.run(tf.global_variables_initializer())
#    num_examples = len(X_train_gray_s)
#    
#    print("Training...")
#    #print()
#    for i in range(EPOCHS):
#        #already shuffled
#        #X_train, y_train = shuffle(X_train, y_train)
#        if use_normalized_train:
#            X_train_gray_norm, y_train_s = shuffle(X_train_gray_norm, y_train_s)
#        else:
#            X_train_gray_s, y_train_s = shuffle(X_train_gray_s, y_train_s)
#               
#        for offset in range(0, num_examples, BATCH_SIZE):
#            end = offset + BATCH_SIZE
#            #batch_x, batch_y = X_train[offset:end], y_train[offset:end]
#            if use_normalized_train:
#                batch_x, batch_y = X_train_gray_norm[offset:end], y_train_s[offset:end]
#                sess.run(training_operation, feed_dict={x: batch_x, y: batch_y,keep_prob:drop_prob_train})
#            else:
#                batch_x, batch_y = X_train_gray_s[offset:end], y_train_s[offset:end]
#                sess.run(training_operation, feed_dict={x: batch_x, y: batch_y,keep_prob:drop_prob_train})
#                
#        
#        if use_normalized_validation:
#            validation_accuracy = evaluate(X_valid_gray_norm, y_valid)
#        else:
#            validation_accuracy = evaluate(X_valid_gray, y_valid)
#        
#        
#        print("EPOCH {} ...".format(i+1))
#        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
#        print()
#        
#    saver.save(sess, './lenet_german_traffic_sign_gray')
#    print("Model saved")
