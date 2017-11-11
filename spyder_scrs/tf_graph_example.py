# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 15:13:19 2017

@author: atpandey
"""

import numpy as np
import tensorflow as tf
graph = tf.Graph()
m1 = np.array([[1.,2.], [3.,4.], [5.,6.], [7., 8.]], dtype=np.float32)
with graph.as_default():
    # Input data.
    m1_input = tf.placeholder(tf.float32, shape=[4,2]) 
    # Ops and variables pinned to the CPU because of missing GPU implementation
# with tf.device('/cpu:0'):
    m2 = tf.Variable(tf.random_uniform([2,3], -1.0, 1.0))
    m3 = tf.matmul(m1_input, m2)
    # This is an identity op with the side effect of printing data when evaluating.
    m3 = tf.Print(m3, [m3], message="m3 is: ")
    # Add variable initializer.
    init = tf.initialize_all_variables()

with tf.Session(graph=graph) as session:
     # We must initialize all variables before we use them.
     init.run()
     print("Initialized")
     print("m2: {}".format(m2))
     print("eval m2: {}".format(m2.eval()))
     feed_dict = {m1_input: m1}
     result = session.run([m3], feed_dict=feed_dict)
     print("\nresult: {}\n".format(result))
