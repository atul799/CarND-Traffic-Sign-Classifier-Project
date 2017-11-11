# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 22:40:39 2017

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
print("mean train:",np.mean(X_train))
print("mean valid:",np.mean(X_valid))
print("mean test:",np.mean(X_test))

#Normalize
X_train=(X_train-128.)/128
X_valid=(X_valid-128.)/128

print("mean train:",np.mean(X_train))
print("mean valid:",np.mean(X_valid))
print("mean test:",np.mean(X_test))
#X_train=(X_train-128)/128
        

#%%
##############
#Preprocess data by shuffling training data

from sklearn.utils import shuffle

X_train, y_train = shuffle(X_train, y_train)
X_valid, y_valid = shuffle(X_valid, y_valid)

#%%
#####################
#Setup tensor flow
import tensorflow as tf

EPOCHS = 10
#EPOCHS = 30

#EPOCHS = 5
BATCH_SIZE = 128

######################

##################
###Convolution network


from tensorflow.contrib.layers import flatten


def conv2d(x, W, b, strides=1,padarg='SAME',name='conv'):
    with tf.name_scope(name):
        x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding=padarg)
        x = tf.nn.bias_add(x, b)
        act=tf.nn.relu(x)
        tf.summary.histogram('wc',W)
        tf.summary.histogram('bc',b)
        tf.summary.histogram('actc',act)
        return act


def maxpool(x, k=2,stride=2,padarg='SAME'):
    return tf.nn.max_pool(
        x,
        ksize=[1, k, k, 1],
        strides=[1,stride,stride, 1],
        padding=padarg)

def fc_layer(x,W,b,name='fc'):
    with tf.name_scope(name):
        x = tf.add(tf.matmul(x, W), b)
        #Activation
        act = tf.nn.relu(x)
        tf.summary.histogram('wfc',W)
        tf.summary.histogram('bfc',b)
        tf.summary.histogram('actfc',act)
        return act

def fc_logit(x,W,b,name='fc'):
    with tf.name_scope(name):
        x = tf.add(tf.matmul(x, W), b)
        #Activation
        #act = tf.nn.relu(x)
        tf.summary.histogram('wfc',W)
        tf.summary.histogram('bfc',b)
        #tf.summary.histogram('actfc',act)
        return x
#n_classes=10

#weights nad biases
#################################################################################
mu = 0
sigma = 0.1
#weights = {
#'wc1': tf.Variable(tf.random_normal([5, 5, 3, 6], mean = mu, stddev = sigma)),
#'wc2': tf.Variable(tf.random_normal([5, 5, 6, 16],mean = mu, stddev = sigma)),
#'wd1': tf.Variable(tf.random_normal([5*5*16, 120],mean = mu, stddev = sigma)),
#'wd2': tf.Variable(tf.random_normal([120, 84],mean = mu, stddev = sigma)),
#'out': tf.Variable(tf.random_normal([84, n_classes],mean = mu, stddev = sigma))}
#
#biases = {
#'bc1': tf.Variable(tf.random_normal([6])),
#'bc2': tf.Variable(tf.random_normal([16])),
#'bd1': tf.Variable(tf.random_normal([120])),
#'bd2': tf.Variable(tf.random_normal([84])),
#'out': tf.Variable(tf.random_normal([n_classes]))}

weights = {
'wc1': tf.Variable(tf.truncated_normal([5, 5, 3, 6], mean = mu, stddev = sigma),name='wc1'),
'wc2': tf.Variable(tf.truncated_normal([5, 5, 6, 16],mean = mu, stddev = sigma),name='wc2'),
'wd1': tf.Variable(tf.truncated_normal([5*5*16, 120],mean = mu, stddev = sigma),name='wd1'),
'wd2': tf.Variable(tf.truncated_normal([120, 84],mean = mu, stddev = sigma),name='wd21'),
'out': tf.Variable(tf.truncated_normal([84, n_classes],mean = mu, stddev = sigma),name='wout')}

biases = {
'bc1': tf.Variable(tf.truncated_normal([6], mean = mu, stddev = sigma),name='bc1'),
'bc2': tf.Variable(tf.truncated_normal([16], mean = mu, stddev = sigma),name='bc2'),
'bd1': tf.Variable(tf.truncated_normal([120], mean = mu, stddev = sigma),name='bd1'),
'bd2': tf.Variable(tf.truncated_normal([84], mean = mu, stddev = sigma),name='bd2'),
'out': tf.Variable(tf.truncated_normal([n_classes], mean = mu, stddev = sigma),name='bout')}

###########################################################################################

# tf Graph input
#x = tf.placeholder(tf.float32, [None, 32, 32, 1])
#y = tf.placeholder(tf.float32, [None, n_classes])
#keep_prob = tf.placeholder(tf.float32)


def LeNet(x):  
    
    #print("shape of X is:",x.shape)
    #print("lenetX",x[0,19:21,14:20,0])
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    #mu = 0
    #sigma = 0.1
    #Data needs regularization? X-mu/sigma ??
    #the random samples for weights needs to be mean mu and std of sigma


    #tf.summary.histogram('wc1',weights['wc1'])
#tf.summary.histogram('wc2',weights['wc2'])
#tf.summary.histogram('wd1',weights['wd1'])
#tf.summary.histogram('wd2',weights['wd2'])
#tf.summary.histogram('wout',weights['out'])
#tf.summary.histogram('bc1',biases['bc1'])
#tf.summary.histogram('bc2',biases['bc2'])
#tf.summary.histogram('db1',biases['bd1'])
#tf.summary.histogram('bd2',biases['bd2'])
#tf.summary.histogram('bout',biases['out'])
    
    # TODO: Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    
    conv1=conv2d(x, weights['wc1'], biases['bc1'], 1,'VALID','conv1')
    #conv1=conv2d(x, weights['wc1'], biases['bc1'], 1,'VALID')
    
    
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

    #conv1 = tf.nn.dropout(conv1, keep_prob)

    # TODO: Layer 2: Convolutional. Output = 10x10x16.
    conv2=conv2d(conv1, weights['wc2'], biases['bc2'], 1,'VALID','conv2')
    #conv2=conv2d(conv1, weights['wc2'], biases['bc2'], 1,'VALID')
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
    #conv2 = tf.nn.dropout(conv2, keep_prob)
    
    
    # TODO: Flatten. Input = 5x5x16. Output = 400.
    #print("WW1",weights['wd1'].get_shape().as_list()[0])
    #print("WW2",weights['wd1'].get_shape().as_list())
    
    #fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1=flatten(conv2)
    #print("WW3",fc1.get_shape().as_list())
    
    # TODO: Layer 3: Fully Connected. Input = 400. Output = 120.
    #fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1=fc_layer(fc1,weights['wd1'], biases['bd1'],'fc1')
    
    # TODO: Activation.
    #fc1 = tf.nn.relu(fc1)
    
    fc1 = tf.nn.dropout(fc1, keep_prob)
    # TODO: Layer 4: Fully Connected. Input = 120. Output = 84.
    #fc2 = tf.add(tf.matmul(fc1, weights['wd2']), biases['bd2'])
    fc2=fc_layer(fc1,weights['wd2'], biases['bd2'],'fc2')
    
    # TODO: Activation.
    #fc2=tf.nn.relu(fc2)
    
    
    fc2 = tf.nn.dropout(fc2, keep_prob)
    
    # TODO: Layer 5: Fully Connected. Input = 84. Output = 10.
    #logits = tf.add(tf.matmul(fc2, weights['out']), biases['out'])
    logits=fc_logit(fc2,weights['out'], biases['out'],'logits')
    #print('WW4',logits.get_shape().as_list())
    return logits
###############################################################################
#Features and labels
x = tf.placeholder(tf.float32, (None, 32, 32, 3),name='x')
y = tf.placeholder(tf.int32, (None),name='labels')
one_hot_y = tf.one_hot(y, 43)
keep_prob=tf.placeholder(tf.float32)

##############################


################3
##Training Pipeline
rate = 0.001

logits = LeNet(x)

with tf.name_scope('cross_entropy'):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
with tf.name_scope('training'):
    loss_operation = tf.reduce_mean(cross_entropy)
    optimizer = tf.train.AdamOptimizer(learning_rate = rate)
    training_operation = optimizer.minimize(loss_operation)
    tf.summary.scalar('cross_entropy',cross_entropy)


#################
#Model Evaluation
with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
    accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))



#input image summary
tf.summary.image('input_imgs',X_train,3)
saver = tf.train.Saver()

#


def evaluate(X_data, y_data,drop_prob_valid,EPOCH,name='valid_eval'):
    with tf.name_scope(name):
        num_examples = len(X_data)
        total_accuracy = 0
        sess = tf.get_default_session()
        #l=0
        for offset in range(0, num_examples, BATCH_SIZE):
            batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
            accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y,keep_prob:drop_prob_valid})
            #if l % 500==0:
            #    print("step %d, valid accuracy is : %g" %(offset,accuracy))
            total_accuracy += (accuracy * len(batch_x))
            #l += 1
            tf.summary.scalar('valid_accuracy',accuracy)
            #s=sess.run(merged_summary,feed_dict={x: batch_x, y: batch_y,keep_prob:drop_prob_train})
            #writer.add_summary(s,EPOCH)
        return total_accuracy / num_examples

##############################
#Train the model
drop_prob_train=0.5
drop_prob_valid=1.0

#X_train=(X_train-128)/128
#X_valid=(X_valid-128)/128

         
X_train_mean=np.mean(X_train)
X_train_std= np.std(X_train)        
X_train=(X_train-X_train_mean)/X_train_std
X_valid=(X_valid-X_train_mean)/X_train_std       

merged_summary=tf.summary.merge_all()



train_accuracy=[]
valid_acc=[]
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)
    print("num_Examples:",num_examples)
    print("Batch size",BATCH_SIZE)
    
    print("Training...")
    print()
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            #hist=sess.run(merged, feed_dict={x: batch_x, y: batch_y,keep_prob:drop_prob_train})
            #writer.add_summary(hist)
            #with tf.name_scope('training'):
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y,keep_prob:drop_prob_train})
            #######
            #if i % 2==0 and offset==num_examples-1:
            if end >= num_examples:
                train_acc=sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y,keep_prob:drop_prob_train})
                train_accuracy.append(train_acc)
                tf.summary.scalar('train_accuracy',train_acc)
                #s=sess.run(merged_summary,feed_dict={x: batch_x, y: batch_y,keep_prob:drop_prob_train})
                #writer.add_summary(s,i)
                #print("step %d " %(i))
                #######
        validation_accuracy = evaluate(X_valid, y_valid,drop_prob_valid,i,'validation')
        valid_acc.append(validation_accuracy)
        print("EPOCH {} ...".format(i+1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()
        #summ = sess.run(merged, global_step=i)


        
    
    writer=tf.summary.FileWriter('./nn_logs1') 
    writer.add_graph(sess.graph)
    saver.save(sess, './lenet_local')
    print("Model saved")
#%%
import matplotlib.pyplot as plt

for j in train_accuracy:
    print("train accuracies were",j)
for j in   valid_acc:
    print("valid accuracies were",j)


t = list(np.linspace(0,len(train_accuracy),10))
plt.plot(t,train_accuracy,label='train')
plt.plot(t,valid_acc,label='valid')
plt.legend()
plt.grid()
plt.show()


##################################
###Evaluate the model
#with tf.Session() as sess:
#    saver.restore(sess, tf.train.latest_checkpoint('.'))
#
#    test_accuracy = evaluate(X_test, y_test)
#    print("Test Accuracy = {:.3f}".format(test_accuracy))
    
#%%
#12/32/240/168/n_class in hidden units
#Run1, lenet,color nonnorm,epoch=10,learnrate1=0.001, no dropout 
#0.923
#Run2, lenet,color nonormalized,epoch=10,learnrate=0.001, dropout=0.5 on FC layers
#0.965
#Run3, lenet,color normalized,epoch=10,learnrate=0.001, no dropout on FC layers
#0.934 
#Run4, lenet,color normalized,epoch=10,learnrate=0.001, dropout=0.5 on FC layers
#0.0.953   
#Run5, lenet,color normalized,epoch=10,learnrate=0.001, dropout on all hidden layers layers
# 0.921   
#Run6, lenet,color normalized,epoch=40,learnrate=0.001, dropout on all hidden layers layers
#0.950   

#####
#with 6/16/120/84/n_class #run5 accuracy is 0.799
#run4 0.935
#
#%%
#import matplotlib.pyplot as plt
#img11=np.array(X_train[18005],dtype=np.float32)
#img11=img11
#print("img11 shape",img11.shape)
#W11=weights['wc1']
#print("W11 shape",W11.shape)
#W22=weights['wc2']
#print("W22 shape",W22.shape)
#
##convl1=np.dot(img1,W1)
##32x32x3,5,5,3,6
##print(convl1.shape)
#nr_imgs1=W11.shape[3]
#nr_imgs2=W22.shape[3]
#
#n_rows = 3
#n_cols = 2
#offset = 9000
#
#plt.imshow(img11)
#img1=img11.reshape(-1,32,32,3)
#print("img1 shape",img1.shape)
#
#fig, axs = plt.subplots(n_rows,n_cols, figsize=(8, 8))
#fig.subplots_adjust(hspace = .1, wspace=.001)
#axs = axs.ravel()
#
#with tf.Session() as sess:
#        sess.run(tf.global_variables_initializer())
#        #W1=tf.reshape(W11[:,:,:,i],(5*5,3))
#        #print(W11[:,:,:,i].shape)
#        #print("W1 shape",W1.shape)
#        img_p=sess.run(tf.nn.conv2d(img1, W11, strides=[1, 1, 1, 1], padding="SAME"))
#        #axs[i].axis('off')
#        img_p1=np.squeeze(img_p)
#        print("img_p1 shape",img_p1.shape)
#        img_p2=sess.run(tf.nn.conv2d(img_p, W22, strides=[1, 1, 1, 1], padding="SAME"))
#        img_p3=np.squeeze(img_p2)
#        print("img_p3 shape",img_p3.shape)
#        #plt.imshow(img_p)
#for i in range(img_p1.shape[2]):
#    axs[i].imshow(img_p1[:,:,i])
#
##%%
#
#fig1, axs1 = plt.subplots(4,4, figsize=(15, 15))
#fig1.subplots_adjust(hspace = .1, wspace=.001)
#axs1 = axs1.ravel()
#for i in range(img_p3.shape[2]):
#    axs1[i].imshow(img_p3[:,:,i])
