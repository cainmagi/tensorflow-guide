#!/usr/python
# -*- coding: UTF8-*- #
'''
####################################################
# Non-linear regression - Tools
# Yuchen Jin @ cainmagi@gmail.com
# Requirements: (Pay attention to version)
#   python 3.6
#   tensorflow r1.13+
#   numpy, matplotlib
# Tools for post-processing and analyzing data which
# is from the output of our non-linear regression 
# model.
# Version: 1.05 # 2019/3/17
# Comments:
#   Adapt it for non-linear regression testing.
# Version: 1.00 # 2019/3/12
# Comments:
#   Create this project.
####################################################
'''

import numpy as np
import matplotlib.pyplot as plt
import os, sys
os.chdir(sys.path[0])

from basic_units import radians

MARKERS = ['o', 'v', 's', 'd', 'x', '+', '>', 'p', '*']

def parseData(path, keys):
    keys_list = dict((k, []) for k in keys)
    name_list = []
    for f in os.scandir(path):
        if f.is_file():
            name, _ = os.path.splitext(f.name)
            name_list.append(name.replace('_', ' '))
            data = np.load(os.path.join(path, f.name))
            for key in keys:
                keys_list[key].append(data[key])
            epoch = data['epoch']
    return name_list, epoch, keys_list
    
def showCurves(path, prefix='{0}', converter=str):
    '''
    Show curves from different tests in a same folder.
    '''
    name_list, epoch, keys_list = parseData(path, ['loss', 'corr'])
    loss_list = keys_list['loss']
    corr_list = keys_list['corr']
    if (not loss_list) or (not corr_list):
        raise FileExistsError('No data found, could not draw curves.')
    
    for i in range(len(loss_list)):
        plt.semilogy(loss_list[i], label=prefix.format(converter(name_list[i])), marker=MARKERS[i%9], markevery=3)
    plt.legend()
    plt.xlabel('epoch'), plt.ylabel('MSE')
    plt.gcf().set_size_inches(5, 5), plt.tight_layout(), plt.show()
    
    for i in range(len(corr_list)):
        plt.plot(corr_list[i], label=prefix.format(converter(name_list[i])), marker=MARKERS[i%9], markevery=3)
    plt.legend()
    plt.xlabel('epoch'), plt.ylabel('Pearson\'s correlation')
    plt.gcf().set_size_inches(5, 5), plt.tight_layout(), plt.show()
    
def showBars(path, prefix='{0}', converter=str, ylim=None):
    '''
    Show bar graphs for RMSE for each result
    '''
    name_list, epoch, keys_list = parseData(path, ['test_y', 'pred_y'])
    #print(keys_list)
    ytrue_list = keys_list['test_y']
    ypred_list = keys_list['pred_y']
    def RMSE(y_true, y_pred):
        return np.sqrt(np.mean(np.mean(np.square(y_true - y_pred), axis=1), axis=1))
    N = ytrue_list[0].shape[0]
    NG = len(ytrue_list)
    for i in range(NG):
        plt.bar([0.6+j+0.8*i/NG+0.4/NG for j in range(-1,ytrue_list[i].shape[0]-1,1)], RMSE(ytrue_list[i], ypred_list[i]), width=0.8/NG, label=prefix.format(converter(name_list[i])))
    plt.legend(ncol=5)
    plt.xlabel('sample'), plt.ylabel('RMSE')
    if ylim is not None:
        plt.ylim([0,ylim])
    plt.gcf().set_size_inches(12, 5), plt.tight_layout(), plt.show()
    
def showData(path, key, ref, prefix='{0}', converter=str, ylabel=None):
    '''
    Show data from different tests in a same folder.
    '''
    name_list, epoch, keys_list = parseData(path, [key, ref])
    key_list = keys_list[key]
    ref_list = keys_list[ref]
    if (not key_list) or (not ref_list):
        raise FileExistsError('No data found, could not draw curves.')
    key_list.append(ref_list[0])
    name_list.append('ref')
    
    for i in range(len(key_list)):
        if name_list[i] == 'ref':
            leglab = 'Ground truth'
        else:
            leglab = prefix.format(converter(name_list[i]))
        plt.plot(key_list[i].ravel(), label=leglab, marker=MARKERS[i%9], markevery=3)
    plt.legend()
    if ylabel is not None:
        plt.ylabel(ylabel)
    plt.gca().xaxis.set_ticklabels([]) 
    plt.gcf().set_size_inches(5, 5), plt.tight_layout(), plt.show()
    
def showFreqDom(path, prefix='{0}', converter=str):
    '''
    Show data from different tests in a same folder.
    '''
    name_list, epoch, keys_list = parseData(path, ['test_y', 'pred_y'])
    yt_list = keys_list['test_y']
    y_list = keys_list['pred_y']
    if (not yt_list) or (not y_list):
        raise FileExistsError('No data found, could not draw curves.')
        
    def calAmp(data):
        return np.abs(data[...,0]+1j*data[...,1]).ravel()
    def calPha(data):
        return [val*radians for val in np.angle(data[...,0]+1j*data[...,1]).ravel()]
    
    for i in range(len(yt_list)):
        titlab = prefix.format(converter(name_list[i]))
        x = [val*radians for val in np.linspace(0, np.pi, yt_list[i].shape[1])]
        plt.plot(x, calAmp(y_list[i][-1,...]), xunits=radians, label='Prediction', linewidth=1)
        plt.plot(x, calAmp(yt_list[i][-1,...]), xunits=radians, label='Ground truth', linewidth=1)
        plt.legend(), plt.xlabel('x'), plt.ylabel('|Y|'), plt.title(titlab)
        plt.gcf().set_size_inches(6, 4), plt.tight_layout(), plt.show()
        
        plt.plot(x, calPha(y_list[i][-1,...]), xunits=radians, yunits=radians, label='Prediction', linewidth=1)
        plt.plot(x, calPha(yt_list[i][-1,...]), xunits=radians, yunits=radians, label='Ground truth', linewidth=1)
        plt.legend(), plt.xlabel('x'), plt.ylabel('∠ Y'), plt.title(titlab)
        plt.gcf().set_size_inches(6, 4), plt.tight_layout(), plt.show()

if __name__ == '__main__':
    showCurves('./test', prefix='L={0}', converter=int)
    showBars('./test', prefix='L={0}', converter=int)
    showData('./test', 'W', 'omega', ylabel='Predicted ω', prefix='L={0}', converter=int)
    showData('./test', 'p', 'phi', ylabel='Predicted φ', prefix='L={0}', converter=int)
    showData('./test', 'b', 'a', ylabel='Predicted a', prefix='L={0}', converter=int)
    showFreqDom('./test', prefix='L={0}', converter=int)