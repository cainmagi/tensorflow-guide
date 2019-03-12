#!/usr/python
# -*- coding: UTF8-*- #
'''
####################################################
# Linear regression - Data parser
# Yuchen Jin @ cainmagi@gmail.com
# Requirements: (Pay attention to version)
#   python 3.6
#   tensorflow r1.13+
#   numpy, matplotlib
# A project for generating data.
# Use logistic regression to learn the best W,b for
#    y ~ W x + b.
# Version: 1.12 # 2019/3/9
# Comments:
#   1. Add a function for low rank matrix
#      approimation.
#   2. Adapt the TestDataSet to TestDataRegSet so
#      that it would genrate raw values. 
# Version: 1.10 # 2019/3/9
# Comments:
#   Remove the sigmoid function in this file.
# Version: 1.00 # 2019/3/4
# Comments:
#   Create this project.
####################################################
'''

#import tensorflow as tf
import numpy as np
#import matplotlib.pyplot as plt

def gen_lowrank(A, r):
    '''
    Generate a low rank approximation to matrix A.
        A: input matrix.
        r: output rank.
    '''
    sze = A.shape
    r_min = np.amin(sze)
    assert r <= r_min and r > 0, 'r should in the range of [1, {0}]'.format(r_min)
    u, s, v = np.linalg.svd(A, full_matrices=False)
    s = np.diag(s[:r])
    return np.matmul(np.matmul(u[:,:r], s), v[:r,:])

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
        self.config(train=True, batch=100, noise=0.1)
        
    def config(self, train=None, batch=None, noise=None):
        '''
        Configuration
        train: a flag for controlling the iterator mode.
        batch: the number of samples in a batch
        noise: std. of the error added to the y.
        '''
        if train is not None:
            self.train = bool(train)
        if batch is not None:
            self.batch = batch
        if noise is not None:
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
            
class TestDataRegSet(TestDataSet):
    '''
    A generator of the data set for testing the linear regression model.
    '''
    def next_train(self):
        '''
        Get the next train batch: (x, y)
        '''
        x = self.s_x * (np.random.random([self.batch, self.len_x]) - 0.5)
        y = np.matmul(x, self.A) + self.c
        if self.noise > 1e-3:
            y = y + np.random.normal(0, self.noise, size=y.shape)
        else:
            np.random.normal(0, self.noise, size=y.shape)
        return x, y

if __name__ == '__main__':

    def test_lowrank():
        A = np.random.normal(0, 10, [10,6])
        for r in range(1,7):
            A_ = gen_lowrank(A, r)
            RMS = np.sqrt(np.mean(np.square(A - A_)))
            R = np.linalg.matrix_rank(A_)
            print('Rank = {0}, RMS={1}'.format(R, RMS))
        
    def test_dataset():
        A = np.random.normal(0, 10, [10,6])
        c = np.random.uniform(1, 3, [1,6])
        dataSet = TestDataSet(10, A, c)
        dIter = iter(dataSet)
        for i in range(10):
            x, y = next(dIter)
            print(np.sum(y,axis=0)/100)
    
    test_lowrank()