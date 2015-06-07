#-*- coding:utf-8 -*-
from base import *
from sklearn.naive_bayes import MultinomialNB,GaussianNB,BernoulliNB

if __name__ == '__main__':
    X,Y = train_sets()
    measure(MultinomialNB,{},X,Y,'NB') 
    #measure(GaussianNB,{},X,Y,'NB-Gaussian') 
    #gen_submission(GaussianNB,{},X,Y,'NB-Gaussian') 
    #bias_variance_analysis(GaussianNB,{},X,Y,'NB-Gaussian') 
    #gen_submission(MultinomialNB,{},X,Y,'NB') 
    #bias_variance_analysis(MultinomialNB,{},X,Y,'NB') 
