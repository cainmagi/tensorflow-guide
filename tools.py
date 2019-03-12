#!/usr/python
# -*- coding: UTF8-*- #
'''
####################################################
# Linear regression - Tools
# Yuchen Jin @ cainmagi@gmail.com
# Requirements: (Pay attention to version)
#   python 3.6
#   tensorflow r1.13+
#   numpy, matplotlib
# Tools for post-processing and analyzing data which
# is from the output of our linear regression model.
# Version: 1.00 # 2019/3/12
# Comments:
#   Create this project.
####################################################
'''

import numpy as np
import matplotlib.pyplot as plt
import os, sys
os.chdir(sys.path[0])

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
        return np.sqrt(np.mean(np.square(y_true - y_pred), axis=1))
    N = ytrue_list[0].shape[0]
    NG = len(ytrue_list)
    for i in range(NG):
        plt.bar([0.6+j+0.8*i/NG+0.4/NG for j in range(-1,9,1)], RMSE(ytrue_list[i], ypred_list[i]), width=0.8/NG, label=prefix.format(converter(name_list[i])))
    plt.legend(ncol=5)
    plt.xlabel('sample'), plt.ylabel('RMSE')
    if ylim is not None:
        plt.ylim([0,ylim])
    plt.gcf().set_size_inches(12, 5), plt.tight_layout(), plt.show()
    
def saveResults(path, opath, oprefix, datakeys, title='', xlabel=None, ylabel=None, onlyFirst=False, plot=False, prefix=' ({0})', converter=str):
    '''
    Save result graphs to a folder.
    '''
    name_list, _, data_list = parseData(path, datakeys)
    if plot: # show curves
        c_list = data_list['c']
        b_list = data_list['b']
        NG = len(b_list)
        for i in range(NG):
            plt.plot(c_list[i].T, label='c')
            plt.plot(b_list[i].T, label='b')
            plt.legend()
            plt.gca().set_title(title+prefix.format(converter(name_list[i])))
            if xlabel is not None:
                plt.xlabel(xlabel)
            if ylabel is not None:
                plt.ylabel(ylabel)
            plt.gcf().set_size_inches(5, 5)
            if onlyFirst:
                formatName = ''
            else:
                formatName = name_list[i].replace(' ', '_')
            plt.savefig(os.path.join(opath, oprefix+'{0}.svg'.format(formatName)))
            plt.close(plt.gcf())
            if onlyFirst:
                return
    else: # show images
        data_list = data_list[datakeys[0]]
        NG = len(data_list)
        for i in range(NG):
            plt.imshow(data_list[i], interpolation='nearest', aspect='auto'), plt.colorbar(),
            plt.gca().set_title(title+prefix.format(converter(name_list[i])))
            if xlabel is not None:
                plt.xlabel(xlabel)
            if ylabel is not None:
                plt.ylabel(ylabel)
            plt.gcf().set_size_inches(6, 5)
            if onlyFirst:
                formatName = ''
            else:
                formatName = name_list[i].replace(' ', '_')
            plt.savefig(os.path.join(opath, oprefix+'{0}.png'.format(formatName)))
            plt.close(plt.gcf())
            if onlyFirst:
                return

if __name__ == '__main__':
    showCurves('./test/algorithm')
    showCurves('./test/noise', prefix='ε=N(0,{0})', converter=int)
    showBars('./test/algorithm', ylim=70)
    showBars('./test/noise', prefix='ε=N(0,{0})', converter=int)

    def saveAllResults():
        saveResults('./test/algorithm', './record/algorithm', 'alg-A-', ['A'], title='A', prefix='', onlyFirst=True)
        saveResults('./test/algorithm', './record/algorithm', 'alg-yt-', ['test_y'], title='True values', prefix='', onlyFirst=True)
        saveResults('./test/algorithm', './record/algorithm', 'alg-y-', ['pred_y'], title='Predicted values')
        saveResults('./test/algorithm', './record/algorithm', 'alg-W-', ['W'], title='W')
        saveResults('./test/algorithm', './record/algorithm', 'alg-cb-', ['c', 'b'], title='Biases', plot=True)
        saveResults('./test/noise', './record/noise', 'noi-A-', ['A'], title='A', prefix='', onlyFirst=True)
        saveResults('./test/noise', './record/noise', 'noi-yt-', ['test_y'], title='True values', prefix=' (ε=N(0,{0}))', converter=int)
        saveResults('./test/noise', './record/noise', 'noi-y-', ['pred_y'], title='Predicted values', prefix=' (ε=N(0,{0}))', converter=int)
        saveResults('./test/noise', './record/noise', 'noi-W-', ['W'], title='W', prefix=' (ε=N(0,{0}))', converter=int)
        saveResults('./test/noise', './record/noise', 'noi-cb-', ['c', 'b'], title='Biases', plot=True, prefix=' (ε=N(0,{0}))', converter=int)
    
    #saveAllResults()