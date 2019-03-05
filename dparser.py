#!/usr/python
# -*- coding: UTF8-*- #
'''
####################################################
# Linear classification - Data parser
# Yuchen Jin @ cainmagi@gmail.com
# Requirements: (Pay attention to version)
#   python 3.6
#   tensorflow r1.12+
#   numpy, matplotlib
# A project for generating data.
# Use logistic regression to learn the best W,b for
#    y ~ W x + b.
# Version: 1.00 # 2019/3/4
# Comments:
#   Create this project.
####################################################
'''

#import tensorflow as tf
import numpy as np
#import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class TestDataSet:
    '''
    A generator of the data set for testing the linear model.
    '''
    def __init__(self, scale_x, A, c):
        '''
        Initialize the data generator.
        scale_x: the scale of input vector.
        A, c: the linear transformation.
        '''
        self.s_x = 2 * scale_x
        self.A = A
        self.c = c
        self.len_x = A.shape[0]
        self.config()
        
    def config(self, train=True, batch=100, noise=0.1):
        '''
        Configuration
        train: a flag for controlling the iterator mode.
        batch: the number of samples in a batch
        noise: std. of the error added to the y.
        '''
        self.train = bool(train)
        self.batch = batch
        self.noise = noise
        
    def next_train(self):
        '''
        Get the next train batch: (x, y)
        '''
        x = self.s_x * (np.random.random([self.batch, self.len_x]) - 0.5)
        y = np.matmul(x, self.A) + self.c
        if self.noise > 1e-3:
            y = y + np.random.normal(0, self.noise, size=y.shape)
        y = np.greater(y, 0.0).astype(np.float32)
        return x, y
    
    def next_test(self):
        '''
        Get the next test batch x.
        '''
        return self.s_x * (np.random.random([self.batch, self.len_x]) - 0.5)
    
    def __iter__(self):
        while True:
            samp = self.__next__()
            if np.any(np.isnan(samp[0])) or np.any(np.isnan(samp[1])):
                print(samp)
                raise ValueError
            yield samp
    
    def __next__(self):
        if self.train:
            return self.next_train()
        else:
            return self.next_test()

if __name__ == '__main__':

    def test_dataset():
        A = np.random.normal(0, 10, [10,6])
        c = np.random.uniform(1, 3, [1,6])
        dataSet = TestDataSet(10, A, c)
        dIter = iter(dataSet)
        for i in range(10):
            x, y = next(dIter)
            print(np.sum(y,axis=0)/100)
    
    test_dataset()