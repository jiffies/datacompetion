#-*- coding:utf-8 -*-
from base import *
from sklearn.lda import LDA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB,GaussianNB
from sklearn.cross_validation import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation

if __name__ == '__main__':
    X,Y = train_sets()
    lr = LogisticRegression()
    gbc = GradientBoostingClassifier(n_estimators=100)
    nb = MultinomialNB()
    knn = neighbors.KNeighborsClassifier(n_neighbors=100)
    
    for clf, label in zip([lr,gbc,nb,knn], ['Logistic Regression', 'Gradient Boost', 'naive Bayes','KNN']):

        scores = cross_validation.cross_val_score(clf, X, Y, cv=3, scoring='roc_auc')
        print("roc_auc: %0.4f (+/- %0.4f) [%s]" % (scores.mean(), scores.std(), label))
