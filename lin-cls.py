#!/usr/python
# -*- coding: UTF8-*- #
'''
####################################################
# Linear classification
# Yuchen Jin @ cainmagi@gmail.com
# Requirements: (Pay attention to version)
#   python 3.6
#   tensorflow r1.13+
#   numpy, matplotlib
# Linear classification demo for Tensroflow.
# Use logistic regression to learn the best W,b for
#    y ~ W x + b.
# Version: 1.10 # 2019/3/9
# Comments:
#   Upgrade the API version from r1.12 to r1.13.
#   The modifications are mainly about definitions
#   of the losses and metrics.
# Version: 1.00 # 2019/3/4
# Comments:
#   Create this project.
####################################################
'''

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import dparser as dp

INPUT_SHAPE = 10
LABEL_SHAPE = 6

class LinClsHandle:
    '''
    Linear classification handle
    A demo of using tf.keras APIs to perform the linear classification.
    '''
    def __init__(self, learning_rate=0.01, epoch=10, steppe=30):
        '''
        Initialization and pass fixed parameters.
            learning_rate: the learning rate for optimizer.
            epoch:         training epochs.
            steppe:        steps per epoch
        '''
        self.lr = learning_rate
        self.epoch = epoch
        self.steppe = steppe
    
    def construct(self):
        '''
        Construct a linear model and set the optimizer as Adam
        '''
        # Construction
        self.model = tf.keras.Sequential()
        self.dense1 = tf.keras.layers.Dense(LABEL_SHAPE, use_bias=True, input_shape=(INPUT_SHAPE,), 
                                  kernel_initializer=tf.keras.initializers.RandomNormal(0.0, stddev=10.0), 
                                  bias_initializer=tf.keras.initializers.Constant(2), 
                                  activation=tf.nn.sigmoid)
        self.model.add(self.dense1)
        
        # Set optimizer
        self.model.compile(
            optimizer=tf.train.AdamOptimizer(self.lr),
            loss=tf.keras.losses.BinaryCrossentropy(), 
            metrics=[self.accuracy, tf.keras.metrics.BinaryAccuracy()]
        )

    @staticmethod
    def accuracy(y_true, y_pred):
        return tf.keras.backend.mean(tf.keras.backend.equal(y_true, tf.keras.backend.round(y_pred)))
    
    def train(self, dataSet):
        '''
        Use a data set to train the network.
        '''
        return self.model.fit(dataSet, epochs=self.epoch, steps_per_epoch=self.steppe)
    
    def test(self, data, labels):
        '''
        Use (data, label) pairs to test the results.
        '''
        loss, accu, binaccu = self.model.evaluate(data, labels)
        print('Evaluated loss (losses.BinaryCrossentropy)  =', loss)
        print('Evaluated accuracy (self defined)           =', accu)
        print('Evaluated accuracy (metrics.BinaryAccuracy) =', accu)
        return self.model.predict(data)
        

if __name__ == '__main__':
    # Initialization
    A = np.random.normal(0, 10, [INPUT_SHAPE, LABEL_SHAPE])
    c = np.random.uniform(1, 3, [1, LABEL_SHAPE])
    dataSet = dp.TestDataSet(10, A, c)
    dataSet.config(batch=32, noise=0.1)
    # Construct the model and train it.
    h = LinClsHandle(learning_rate=0.01, epoch=20, steppe=500)
    h.construct()
    record = h.train(iter(dataSet))
    
    # Show records
    plt.semilogy(record.epoch, record.history['loss']), plt.xlabel('epoch'), plt.ylabel('Cross entropy')
    plt.gcf().set_size_inches(5, 5), plt.show()
    plt.plot(record.epoch, record.history['accuracy'], label='self defined'), plt.plot(record.epoch, record.history['binary_accuracy'], label='from tensorflow'), plt.xlabel('epoch'), plt.ylabel('Accuracy'), plt.legend()
    plt.gcf().set_size_inches(5, 5), plt.show()
    
    # Generate a group of testing samples:
    dataSet.config(batch=10)
    x, y = next(dataSet)
    
    # Check the testing results
    yp = h.test(x, y)
    _, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(y, interpolation='nearest', aspect='auto')
    ax1.set_title('True class')
    ax2.imshow(yp, interpolation='nearest', aspect='auto')
    ax2.set_title('Predicted class')
    plt.gcf().set_size_inches(10, 5), plt.show()
    
    # Check the regressed values
    W, b = h.dense1.get_weights()
    plt.imshow(A, interpolation='nearest', aspect='auto'), plt.colorbar(), plt.gca().set_title('A')
    plt.gcf().set_size_inches(6, 5), plt.show()
    plt.imshow(W, interpolation='nearest', aspect='auto'), plt.colorbar(), plt.gca().set_title('W')
    plt.gcf().set_size_inches(6, 5), plt.show()
    
    plt.plot(c.T, label='c')
    plt.plot(b.T, label='b')
    plt.legend()
    plt.gcf().set_size_inches(5, 5), plt.show()