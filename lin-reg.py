#!/usr/python
# -*- coding: UTF8-*- #
'''
####################################################
# Linear regression
# Yuchen Jin @ cainmagi@gmail.com
# Requirements: (Pay attention to version)
#   python 3.6
#   tensorflow r1.13+
#   numpy, matplotlib
# Linear classification demo for Tensroflow.
# Use linear regression to learn the best W,b for
#    y ~ W x + b.
# Version: 1.00 # 2019/3/12
# Comments:
#   Finish this project. Now it could train and 
#   test with different options.
# Version: 0.50 # 2019/3/9
# Comments:
#   Create this project.
####################################################
'''

import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt

import extension as ext
import dparser as dp
import os, sys
os.chdir(sys.path[0])

INPUT_SHAPE = 10
LABEL_SHAPE = 6
RANK = 2

class LinRegHandle(ext.AdvNetworkBase):
    '''
    Linear regression handle
    A demo of using tf.keras APIs to perform the linear regression.
    '''
    def __init__(self, learning_rate=0.01, epoch=10, steppe=30, optimizerName='adam'):
        '''
        Initialization and pass fixed parameters.
            learning_rate: the learning rate for optimizer.
            epoch:         training epochs.
            steppe:        steps per epoch
            optimizerName: the name of optimizer (available: 'adam', 'amsgrad', 
                           'adamax', 'nadam', 'adadelta', 'rms', 'adagrad', 
                           'nmoment', 'sgd')
        '''
        self.lr = learning_rate
        self.epoch = epoch
        self.steppe = steppe
        self.optimizerName = optimizerName
    
    def construct(self):
        '''
        Construct a linear model and set the optimizer as Adam
        '''
        # Construction
        input = tf.keras.Input(shape=(INPUT_SHAPE,), dtype=tf.float32)
        dense1 = tf.keras.layers.Dense(LABEL_SHAPE, use_bias=True, 
                                  kernel_initializer=tf.keras.initializers.RandomNormal(0.0, stddev=10.0), 
                                  bias_initializer=tf.keras.initializers.Constant(2), 
                                  activation=None, name='dense1')(input)
        self.model = tf.keras.Model(inputs=input, outputs=dense1)
        
        # Set optimizer
        self.model.compile(
            optimizer=self.optimizer(self.optimizerName, self.lr),
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[self.relation]
        )
        
    @staticmethod
    def relation(y_true, y_pred):
        m_y_true = tf.keras.backend.mean(y_true, axis=0)
        m_y_pred = tf.keras.backend.mean(y_pred, axis=0)
        s_y_true = tf.keras.backend.sqrt(tf.keras.backend.mean(tf.keras.backend.square(y_true), axis=0) - tf.keras.backend.square(m_y_true))
        s_y_pred = tf.keras.backend.sqrt(tf.keras.backend.mean(tf.keras.backend.square(y_pred), axis=0) - tf.keras.backend.square(m_y_pred))
        return tf.keras.backend.mean((tf.keras.backend.mean(y_true * y_pred, axis=0) - m_y_true * m_y_pred)/(s_y_true * s_y_pred))
    
    def train(self, dataSet):
        '''
        Use a data set to train the network.
        '''
        return self.model.fit(dataSet, epochs=self.epoch, steps_per_epoch=self.steppe)
    
    def test(self, data, labels):
        '''
        Use (data, label) pairs to test the results.
        '''
        loss, corr = self.model.evaluate(data, labels)
        print('Evaluated loss (losses.MeanSquaredError) =', loss)
        print('Evaluated metric (Pearson\'s correlation) =', corr)
        return self.model.predict(data), loss, corr
        

if __name__ == '__main__':
    import argparse
    
    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Unsupported value encountered.')
    
    parser = argparse.ArgumentParser(
        description='A united version of several kinds of neural networks.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Parse arguments.
    parser.add_argument(
        '-o', '--optimizer', default='adam', metavar='str',
        help='''\
        The optimizer we use to train the model (available: 'adam', 'amsgrad', 'adamax', 
        'nadam', 'adadelta', 'rms', 'adagrad', 'nmoment', 'sgd')
        '''
    )
    
    parser.add_argument(
        '-do', '--outputData', default=None, metavar='str',
        help='''\
        The name of the file where we store the output data for this test. If None, do not output.
        '''
    )
    
    parser.add_argument(
        '-lr', '--learningRate', default=0.01, type=float, metavar='float',
        help='''\
        The learning rate for training the model.
        '''
    )
    
    parser.add_argument(
        '-e', '--epoch', default=20, type=int, metavar='int',
        help='''\
        The number of epochs for training.
        '''
    )
    
    parser.add_argument(
        '-se', '--steppe', default=500, type=int, metavar='int',
        help='''\
        The number of steps per epoch for training.
        '''
    )
    
    parser.add_argument(
        '-tbn', '--trainBatchNum', default=32, type=int, metavar='int',
        help='''\
        The number of samples per batch for training.
        '''
    )
    
    parser.add_argument(
        '-tsn', '--testBatchNum', default=10, type=int, metavar='int',
        help='''\
        The number of samples for testing.
        '''
    )
    
    parser.add_argument(
        '-sd', '--seed', default=None, type=int, metavar='int',
        help='''\
        Seed of the random generaotr. If none, do not set random seed.
        '''
    )
    
    parser.add_argument(
        '-is', '--noise', default=10.0, type=float, metavar='float',
        help='''\
        Standard error of the white noise added to the labels.
        '''
    )
    
    args = parser.parse_args()
    def setSeed(seed):
        np.random.seed(seed)
        random.seed(seed+12345)
        tf.set_random_seed(seed+1234)
    if args.seed is not None: # Set seed for reproductable results
        setSeed(args.seed)
    
    # Initialization
    A = dp.gen_lowrank(np.random.normal(0, 10, [INPUT_SHAPE, LABEL_SHAPE]), RANK)
    c = np.random.uniform(1, 3, [1, LABEL_SHAPE])
    dataSet = dp.TestDataRegSet(10, A, c)
    dataSet.config(noise=args.noise)
    # Generate a group of testing samples.
    if args.seed is not None:
        setSeed(args.seed+1000)
    dataSet.config(batch=args.testBatchNum)
    x, y = next(dataSet)
    # Set the data set for training.
    dataSet.config(batch=args.trainBatchNum)
    # Construct the model and train it.
    h = LinRegHandle(learning_rate=args.learningRate, epoch=args.epoch, 
                     steppe=args.steppe, optimizerName=args.optimizer)
    h.construct()
    print('Begin to train:')
    print('---------------')
    record = h.train(iter(dataSet))
    
    # Generate a group of testing samples:
    dataSet.config(batch=args.testBatchNum)
    x, y = next(dataSet)
    
    # Check the testing results
    print('Begin to test:')
    print('---------------')
    yp, loss_p, corr_p = h.test(x, y)
    
    # Check the regressed values
    W, b = h.model.get_layer(name='dense1').get_weights()
    
    # Save
    if args.outputData is not None:
        np.savez_compressed(args.outputData, 
            epoch = record.epoch,
            loss = record.history['loss'], corr = record.history['relation'], 
            test_x = x, test_y = y, pred_y = yp, 
            pred_loss = loss_p, pred_corr = corr_p,
            W=W, b=b, A=A, c=c
        )