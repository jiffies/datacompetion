#-*- coding:utf-8 -*-
from base import *
from sklearn.linear_model import LogisticRegression

if __name__ == '__main__':
    X,Y = train_sets()
    measure(LogisticRegression,{},X,Y,'LR') 
    #bias_variance_analysis(LogisticRegression,{},X,Y,'LR') 
    #gen_submission(LogisticRegression,{},X,Y,'LR') 

