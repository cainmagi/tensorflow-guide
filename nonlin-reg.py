#!/usr/python
# -*- coding: UTF8-*- #
'''
####################################################
# Non-linear regression
# Yuchen Jin @ cainmagi@gmail.com
# Requirements: (Pay attention to version)
#   python 3.6
#   tensorflow r1.13+
#   numpy, matplotlib
# Perform regression on an analytic non-linear model
# in frequency domain.
# Version: 0.50 # 2019/3/15
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

PARAMS_SHAPE = 10

class NonLinRegHandle(ext.AdvNetworkBase):
    '''
    Non-linear regression handle
    A demo of using tf.keras APIs to perform the non-linear regression.
    '''
    def __init__(self, xLength=100, learningRate=0.01, epoch=10, steppe=30, optimizerName='amsgrad'):
        '''
        Initialization and pass fixed parameters.
            xLength:       the length of the input vector.
            learningRate:  the learning rate for optimizer.
            epoch:         training epochs.
            steppe:        steps per epoch
            optimizerName: the name of optimizer (available: 'adam', 'amsgrad', 
                           'adamax', 'nadam', 'adadelta', 'rms', 'adagrad', 
                           'nmoment', 'sgd')
        '''
        self.lr = learningRate
        self.epoch = epoch
        self.steppe = steppe
        self.optimizerName = optimizerName
        self.xLength = xLength
    
    def construct(self):
        '''
        Construct a linear model and set the optimizer as Adam
        '''
        # Construction
        input = tf.keras.Input(shape=(self.xLength,), dtype=tf.float32)
        upAff = ext.UpDimAffine(PARAMS_SHAPE, use_bias=True, 
                                kernel_initializer = tf.keras.initializers.RandomUniform(minval=0.0, maxval=3.0),
                                bias_initializer = tf.keras.initializers.RandomUniform(minval=0.0, maxval=2.0),
                                kernel_constraint = tf.keras.constraints.NonNeg(),
                                bias_constraint = tf.keras.constraints.NonNeg(),
                                activation=tf.math.cos, name='up_dim_affine')(input)
        dnAff = ext.FFTAffine(name='fft_affine')(upAff)
        
        self.model = tf.keras.Model(inputs=input, outputs=dnAff)
        
        # Set optimizer
        self.model.compile(
            optimizer=self.optimizer(self.optimizerName, self.lr),
            loss=tf.keras.losses.mean_squared_error,
            metrics=[self.relation]
        )
        
        self.model.summary()
        
    @staticmethod
    def relation(y_true, y_pred):
        m_y_true = tf.keras.backend.mean(y_true, axis=0)
        m_y_pred = tf.keras.backend.mean(y_pred, axis=0)
        s_y_true = tf.keras.backend.sqrt(tf.keras.backend.mean(tf.keras.backend.square(y_true), axis=0) - tf.keras.backend.square(m_y_true))
        s_y_pred = tf.keras.backend.sqrt(tf.keras.backend.mean(tf.keras.backend.square(y_pred), axis=0) - tf.keras.backend.square(m_y_pred))
        s_denom = s_y_true * s_y_pred
        s_numer = tf.keras.backend.mean(y_true * y_pred, axis=0) - m_y_true * m_y_pred
        s_index = tf.keras.backend.greater(s_denom, 0)
        return tf.keras.backend.mean(tf.boolean_mask(s_numer,s_index)/tf.boolean_mask(s_denom,s_index))
    
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
        description='Perform regression on an analytic non-linear model in frequency domain.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Parse arguments.
    parser.add_argument(
        '-o', '--optimizer', default='amsgrad', metavar='str',
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
        '-xl', '--xLength', default=100, type=int, metavar='int',
        help='''\
        The length of the input vector.
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
    
    args = parser.parse_args()
    def setSeed(seed):
        np.random.seed(seed)
        random.seed(seed+12345)
        tf.set_random_seed(seed+1234)
    if args.seed is not None: # Set seed for reproductable results
        setSeed(args.seed)
    
    def groupSort(*params):
        sortind = np.argsort(params[0]).flatten()
        res = []
        for p in params:
            if p.shape[0] > p.shape[1]:
                p = p[sortind, :]
            else:
                p = p[:, sortind]
            res.append(p)
        return res
        
    # Initialization
    omega = 3*np.random.random([1, PARAMS_SHAPE])
    phi = 2*np.random.random([1, PARAMS_SHAPE])
    a = np.random.normal(0, 1, [PARAMS_SHAPE, 1])
    dataSet = dp.TestDataFFTSet(3, args.xLength, omega, phi, a)
    # Generate a group of testing samples.
    if args.seed is not None:
        setSeed(args.seed+1000)
    dataSet.config(batch=args.testBatchNum)
    x, y = next(dataSet)
    # Set the data set for training.
    dataSet.config(batch=args.trainBatchNum)
    # Construct the model and train it.
    h = NonLinRegHandle(xLength = args.xLength, learningRate=args.learningRate,
                        epoch=args.epoch, steppe=args.steppe, optimizerName=args.optimizer)
    h.construct()
    print('Begin to train:')
    print('---------------')
    record = h.train(iter(dataSet))
    
    # Generate a group of testing samples:
    dataSet.config(batch=args.testBatchNum)
    x2 = np.reshape(np.linspace(-3,3, args.xLength), [1, args.xLength])
    y2 = dataSet.mapfunc(x2)
    x = np.concatenate([x,x2],axis=0)
    y = np.concatenate([y,y2],axis=0)
    
    # Check the testing results
    print('Begin to test:')
    print('---------------')
    yp, loss_p, corr_p = h.test(x, y)
    
    # Check the regressed values
    w, p = h.model.get_layer(name='up_dim_affine').get_weights()
    b = h.model.get_layer(name='fft_affine').get_weights()[0]
    
    # Resort data
    w, b, p = groupSort(w, b, p) # The solution
    omega, phi, a = groupSort(omega, phi, a) # The ground truth
    
    # Save
    if args.outputData is not None:
        np.savez_compressed(args.outputData, 
            epoch = record.epoch,
            loss = record.history['loss'], corr = record.history['relation'], 
            test_x = x, test_y = y, pred_y = yp, 
            pred_loss = loss_p, pred_corr = corr_p,
            W=w, p=p, b=b, omega=omega, phi=phi, a=a
        )