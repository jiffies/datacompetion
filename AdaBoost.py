#-*- coding:utf-8 -*-
from base import *
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier



if __name__ == '__main__':
    X,Y = train_sets()
    measure(GradientBoostingClassifier,{'n_estimators':100},X,Y,'GradientBoost') 
    #gen_submission(GradientBoostingClassifier,{'n_estimators':400},X,Y,'GradientBoost') 
    #gen_submission(AdaBoostClassifier,{'n_estimators':200},X,Y,'Ada') 
    #bias_variance_analysis(AdaBoostClassifier,{},X,Y,'Ada') 
