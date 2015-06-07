#-*- coding:utf-8 -*-
from base import *
from sklearn.ensemble import RandomForestClassifier


if __name__ == '__main__':
    X,Y = train_sets()
    measure(RandomForestClassifier,{'n_estimators':200},X,Y,'RF') 
    #gen_submission(RandomForestClassifier,{'n_estimators':100},X,Y,'RF') 
    #bias_variance_analysis(RandomForestClassifier,{'n_estimators':100},X,Y,'RF') 
