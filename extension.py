#!/usr/python
# -*- coding: UTF8-*- #
'''
####################################################
# Tensorflow extension
# Yuchen Jin @ cainmagi@gmail.com
# Requirements: (Pay attention to version)
#   python 3.6
#   tensorflow r1.13+
#   numpy, matplotlib
# An extension for Tensorflow APIs. It enables us
# to use more highly integrated APIs such as 
# advanced network layers, special initializers,
# specially designed optimizers and etc.
# Version: 0.10 # 2019/3/11
# Comments:
#   Create this project.
####################################################
'''

import tensorflow as tf

class AdvNetworkBase:
    '''
    Base object of the advanced network APIs.
    '''
    
    @staticmethod
    def optimizer(name='adam', l_rate=0.01, decay=0.0):
        '''
        Define the optimizer by default parameters except learning rate.
        Note that most of optimizers do not suggest users to modify their
        speically designed parameters.
        name: the name of optimizer (default='adam') (available: 'adam', 
              'amsgrad', 'adamax', 'nadam', 'adadelta', 'rms', 'adagrad',
              'nmoment', 'sgd')
        l_rate: learning rate (default=0.01)
        decay: decay ratio ('adadeltaDA' do not support this option)
        '''
        name = name.casefold()
        if name == 'adam':
            return tf.keras.optimizers.Adam(l_rate, decay=decay)
        elif name == 'amsgrad':
            return tf.keras.optimizers.Adam(l_rate, decay=decay, amsgrad=True)
        elif name == 'adamax':
            return tf.keras.optimizers.Adamax(l_rate, decay=decay)
        elif name == 'nadam':
            return tf.keras.optimizers.Nadam(l_rate, schedule_decay=decay)
        elif name == 'adadelta':
            return tf.keras.optimizers.Adadelta(l_rate, decay=decay)
        elif name == 'rms':
            return tf.keras.optimizers.RMSprop(l_rate, decay=decay)
        elif name == 'adagrad':
            return tf.keras.optimizers.Adagrad(l_rate, decay=decay)
        elif name == 'nmoment':
            return tf.keras.optimizers.SGD(lr=l_rate, momentum=0.6, decay=decay, nesterov=True)
        elif name == 'sgd':
            return tf.keras.optimizers.SGD(l_rate, decay=decay)