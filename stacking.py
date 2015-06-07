#-*- coding:utf-8 -*-
from base import *
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB,GaussianNB
from sklearn.cross_validation import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier

class Stacking(object):
    def __init__(self,estimator_dict=[],meta_class=neighbors.KNeighborsClassifier,meta_parameters={'n_neighbors':100}):
        if not estimator_dict:
            return
        self.estimators = []
        for estimator,parameters in estimator_dict.iteritems():
            self.estimators.append(estimator(**parameters))
        self.meta_clf = meta_class(**meta_parameters)

    def __trans_feature(self,x):
        X = self.X
        Y = self.Y
        skf = StratifiedKFold(Y,10)
        blend_test = np.zeros((x.shape[0], len(self.estimators)))
        for j,estimator in enumerate(self.estimators):
            #blend_test_j = np.zeros((x.shape[0], len(skf)))
            #for i,(train,test) in enumerate(skf):
                #print "predict fold: %d" % i
            proba = estimator.predict_proba(x)[:,1]
                #blend_test_j[:,i] = proba
            blend_test[:,j] = proba#blend_test_j.mean(1)
        return blend_test.mean(axis=1).reshape(blend_test.shape[0],1)

    def fit(self,X,Y):
        self.X = X
        self.Y = Y
        #skf = StratifiedKFold(Y,10)
        skf = KFold(n=len(X),n_folds=10,shuffle=True)
        blend_train = np.zeros((X.shape[0], len(self.estimators)))
        for j,estimator in enumerate(self.estimators):
            #print "fit estimator: %d" % j
            #for train,test in skf:
                #x_train,y_train = X[train],Y[train]
                #x_test,y_test = X[test],Y[test]
                estimator.fit(X,Y)
                proba = estimator.predict_proba(X)[:,1]
                blend_train[:, j] = proba
        meta_x = blend_train.mean(axis=1)
        meta_x = meta_x.reshape(meta_x.shape[0],1)
        meta_y = Y
        self.meta_clf.fit(meta_x,meta_y)

    def predict_proba(self,x):
        x = self.__trans_feature(x)
        return self.meta_clf.predict_proba(x)

    def score(self,x,y):
        x = self.__trans_feature(x)
        return self.meta_clf.score(x,y)

if __name__ == '__main__':
    X,Y = train_sets()
    para = {neighbors.KNeighborsClassifier:{'n_neighbors':100},
                    #LogisticRegression:{},
                    #RandomForestClassifier:{'n_estimators':100},
                    #LogisticRegression:{}}
                    GradientBoostingClassifier:{'n_estimators':100},
                    MultinomialNB:{}}
    #measure(Stacking,{'estimator_dict':para,'meta_class':neighbors.KNeighborsClassifier,'meta_parameters':{'n_neighbors':100}},X,Y,'Stacking') 
    measure(Stacking,{'estimator_dict':para,'meta_class':LogisticRegression,'meta_parameters':{}},X,Y,'Stacking') 
    #measure(Stacking,{'estimator_dict':para,'meta_class':MultinomialNB,'meta_parameters':{}},X,Y,'Stacking') 




