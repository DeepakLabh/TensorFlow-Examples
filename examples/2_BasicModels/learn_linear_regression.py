import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

rng = np.random

l_r = 0.01
epochs = 1000
disp_gap = 50

x_train = np.array(map(lambda x:x+(x%3), xrange(15)))
y_train = np.array(map(lambda x:x+2 if x%2==0 else x-3, xrange(15)))
n_samples = x_train.shape[0]

X = tf.placeholder('float')
Y = tf.placeholder('float')

W = tf.Variable(rng.randn(), name='weight')
b = tf.Variable(rng.randn(), name='bias')

pred = tf.add(tf.mul(W,X), b)
cost = tf.reduce_sum(tf.pow(pred-Y, 2))/n_samples
optimizer = tf.train.GradientDescentOptimizer(l_r).minimize(cost)

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    
    for e in range(epochs):
	for (x,y) in zip(x_train, y_train):
	    sess.run(optimizer, feed_dict = {X:x, Y:y})

        if (e+1) % disp_gap == 0:
    	    c = sess.run(cost, feed_dict = {X: x_train, Y:y_train})
    	    print ('Epoch: ', '%04d' % (e+1), 'Loss = ', "{:.9f}".format(c), "W=", sess.run(W), "b=", sess.run(b))
    
    print 'Finished Optimizing'
    
