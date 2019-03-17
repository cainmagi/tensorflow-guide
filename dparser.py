#!/usr/python
# -*- coding: UTF8-*- #
'''
####################################################
# Non-linear regression - Data parser
# Yuchen Jin @ cainmagi@gmail.com
# Requirements: (Pay attention to version)
#   python 3.6
#   tensorflow r1.13+
#   numpy, matplotlib
# A project for generating data.
# Version: 1.15 # 2019/3/16
# Comments:
#   Adapt the TestDataSet to TestDataFFTSet so that
#   it would generate values in non-linear model.
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
        x = self.next_test()
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
        
class TestDataFFTSet(TestDataSet):
    '''
    A generator of the data set for testing the non-linear regression model.
    y = cos(x w^T + 1 p^T) a
    '''
    def __init__(self, scale_x, len_x, omega, phi, a):
        '''
        Initialize the data generator.
        scale_x: the scale of input vector.
        len_x: the length of input vector.
        omega (w) [1 x N]: the inner linear transformation.
        phi (p) [1 x N]: the inner bias.
        a [N x 1]: the outer linear transormation.
        '''
        self.s_x = 2 * scale_x
        self.omega = omega
        self.phi = phi
        self.a = a
        self.len_x = len_x
        self.config(train=True, batch=100, noise=0.0)
        
    def mapfunc(self, x):
        xu = np.expand_dims(x, -1)
        y1 = np.tensordot(xu, self.omega, (2, 0))
        y1 = np.cos(y1 + np.tensordot(np.ones_like(xu), self.phi, (2, 0)))
        y2 = np.squeeze(np.tensordot(y1, self.a, (2, 0)), axis=-1)
        y2 = np.fft.rfft(y2)
        y_r = np.expand_dims(np.real(y2), -1)
        y_i = np.expand_dims(np.imag(y2), -1)
        y = np.concatenate([y_r, y_i], axis=-1)
        return y
        
    def next_train(self):
        '''
        Get the next train batch: (x, y)
        '''
        x = self.next_test()
        y = self.mapfunc(x)
        return x, y

if __name__ == '__main__':

    def test_dataset():
        omega = 3*np.random.random([1, 12])
        phi = 2*np.random.random([1, 12])
        a = np.random.normal(0, 1, [12, 1])
        dataSet = TestDataFFTSet(1, 10, omega, phi, a)
        dIter = iter(dataSet)
        for i in range(10):
            x, y = next(dIter)
            print(y.shape, np.abs(y[0,...,0]+1j*y[0,...,1]))
    
    test_dataset()