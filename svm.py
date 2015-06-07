#-*- coding:utf-8 -*-
from base import *
from sklearn import svm

if __name__ == '__main__':
    X,Y = train_sets()
    #measure(svm.LinearSVC,{'probability':True},X,Y,'svm') 
    #bias_variance_analysis(svm.LinearSVC,{},X,Y,'svm') 
    gen_submission(svm.SVC,{'probability':True},X,Y,'svm') 
