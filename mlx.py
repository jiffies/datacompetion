#-*- coding:utf-8 -*-
from base import *
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB,GaussianNB
from sklearn.cross_validation import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from mlxtend.sklearn import EnsembleClassifier


if __name__ == '__main__':
    X,Y = train_sets()
    lr = LogisticRegression()
    gbc = GradientBoostingClassifier(n_estimators=100)
    knn = neighbors.KNeighborsClassifier(n_neighbors=100)

    measure(EnsembleClassifier,{'clfs':[lr,gbc,knn],'voting':'soft','weights':[1,3,2]},X,Y,'vote')
