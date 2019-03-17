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
# Version: 0.15 # 2019/3/15
# Comments:
#   1. Add two affine transform layers.
#   2. Add a 1-d FFT layer.
# Version: 0.10 # 2019/3/11
# Comments:
#   Create this project.
####################################################
'''

import tensorflow as tf
from tensorflow.python.keras import optimizers
from tensorflow.python.keras import activations
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.engine.input_spec import InputSpec

class FFTAffine(Layer):
    '''
    A layer which applies FFT after a dimension-decreasing affine
    transformation.
    Arguments:
        kernel_initializer: Initializer for the `kernel` weights matrix.
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix.
        kernel_constraint: Constraint function applied to
            the `kernel` weights matrix.
    Input shape:
        nD tensor with shape: `(batch_size, ..., L, M)`.
        The most common situation would be
        a 3D input with shape `(batch_size, L, M)`.
    Output shape:
        nD tensor with shape: `(batch_size, ..., L)`.
        For instance, for a 3D input with shape `(batch_size, L, M)`,
        the output would have shape `(batch_size, (L + 1)//2, 2)`.
    '''
    def __init__(self, kernel_initializer='glorot_uniform', kernel_regularizer=None, kernel_constraint=None, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(FFTAffine, self).__init__(**kwargs)
        
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)

        self.supports_masking = True
        self.input_spec = InputSpec(min_ndim=3)
        
    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        self.layer_Dense = tf.keras.layers.Dense(1, use_bias=False, 
                           kernel_initializer = self.kernel_initializer,
                           kernel_regularizer = self.kernel_regularizer,
                           kernel_constraint = self.kernel_constraint)
        self.layer_Dense.build(input_shape)
        super(FFTAffine, self).build(input_shape)
        
    def call(self, inputs):
        res = tf.squeeze(self.layer_Dense(inputs), -1)
        res = tf.signal.rfft(res)
        res_r = tf.expand_dims(tf.real(res), -1)
        res_i = tf.expand_dims(tf.imag(res), -1)
        res = tf.concat([res_r, res_i], -1)
        return res

    def compute_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        shape_lst = input_shape.as_list()
        if shape_lst[-2] is None:
            return input_shape[:-1].concatenate(2)
        else:
            if shape_lst[-2] % 2 == 0:
                return input_shape[:-2].concatenate(shape_lst[-2]//2+1).concatenate(2)
            else:
                return input_shape[:-2].concatenate((shape_lst[-2]+1)//2).concatenate(2)
        
    def get_config(self):
        config = {
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint)
        }
        base_config = super(FFTAffine, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class UpDimAffine(Layer):
    '''
    The layer that would increase the dimension of a vector by affine
    transformation.
    Arguments:
        units: Positive integer, dimensionality of the output space.
        activation: Activation function to use.
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix.
        bias_initializer: Initializer for the bias vector.
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix.
        bias_regularizer: Regularizer function applied to the bias vector.
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation")..
        kernel_constraint: Constraint function applied to
            the `kernel` weights matrix.
        bias_constraint: Constraint function applied to the bias vector.
    Input shape:
        nD tensor with shape: `(batch_size, ..., input_dim)`.
        The most common situation would be
        a 2D input with shape `(batch_size, input_dim)`.
    Output shape:
        nD tensor with shape: `(batch_size, ..., input_dim, units)`.
        For instance, for a 2D input with shape `(batch_size, input_dim)`,
        the output would have shape `(batch_size, input_dim, units)`.
    '''

    def __init__(self, units, activation=None, use_bias=True,
                 kernel_initializer='glorot_uniform', bias_initializer='zeros',
                 kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
                 kernel_constraint=None, bias_constraint=None, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(UpDimAffine, self).__init__(
            activity_regularizer=regularizers.get(activity_regularizer), **kwargs)
        self.units = int(units)
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.supports_masking = True
        self.input_spec = InputSpec(min_ndim=2)

    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        self.kernel = self.add_weight(
                      'kernel',
                      shape=[1, self.units],
                      initializer=self.kernel_initializer,
                      regularizer=self.kernel_regularizer,
                      constraint=self.kernel_constraint,
                      dtype=self.dtype,
                      trainable=True)
        if self.use_bias:
            self.bias = self.add_weight(
                        'bias',
                        shape=[1, self.units],
                        initializer=self.bias_initializer,
                        regularizer=self.bias_regularizer,
                        constraint=self.bias_constraint,
                        dtype=self.dtype,
                        trainable=True)
        else:
            self.bias = None
        super(UpDimAffine, self).build(input_shape)

    def call(self, inputs):
        inputs = tf.expand_dims(tf.convert_to_tensor(inputs), -1)
        rank = inputs.get_shape().ndims
        res = tf.tensordot(inputs, self.kernel, [[rank - 1], [0]])
        if self.use_bias:
            varbias = tf.tensordot(tf.ones_like(inputs), self.bias, [[rank - 1], [0]])
            res = tf.add(res, varbias)
        if self.activation is not None:
            res = self.activation(res)
        return res

    def compute_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        return input_shape.concatenate(self.units)
        
    def get_config(self):
        config = {
            'units': self.units,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer':
                regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super(UpDimAffine, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

customObjects = {
    'FFTAffine': FFTAffine,
    'UpDimAffine': UpDimAffine
}

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
            return optimizers.Adam(l_rate, decay=decay)
        elif name == 'amsgrad':
            return optimizers.Adam(l_rate, decay=decay, amsgrad=True)
        elif name == 'adamax':
            return optimizers.Adamax(l_rate, decay=decay)
        elif name == 'nadam':
            return optimizers.Nadam(l_rate, schedule_decay=decay)
        elif name == 'adadelta':
            return optimizers.Adadelta(l_rate, decay=decay)
        elif name == 'rms':
            return optimizers.RMSprop(l_rate, decay=decay)
        elif name == 'adagrad':
            return optimizers.Adagrad(l_rate, decay=decay)
        elif name == 'nmoment':
            return optimizers.SGD(lr=l_rate, momentum=0.6, decay=decay, nesterov=True)
        elif name == 'sgd':
            return optimizers.SGD(l_rate, decay=decay)
            
if __name__ == '__main__':

    import numpy as np
    
    def test_layers():
        # Set model and see the summary of the model
        model = tf.keras.models.Sequential([
            UpDimAffine(10, use_bias=True, activation=tf.math.cos, input_shape=(5,)),
            FFTAffine(trainable=False)
        ])
        model.compile(
            optimizer=optimizers.Adam(0.01),
            loss=tf.keras.losses.mean_squared_error,
            metrics=[tf.keras.metrics.mean_squared_error]
        )
        model.summary()
        model.save('my_model.h5')
        # perform the test
        var_input = np.ones([2, 5])
        var_output = model.predict(var_input)
        print(var_input.shape, var_output.shape)
        print(var_output)
    
    def test_read():
        customObjects['cos'] = tf.math.cos
        new_model = tf.keras.models.load_model('my_model.h5', custom_objects=customObjects)
        new_model.summary()
        var_input = np.ones([2, 5])
        var_output = new_model.predict(var_input)
        print(var_input.shape, var_output.shape)
        print(var_output)
    
    test_read()