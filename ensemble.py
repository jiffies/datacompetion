#-*- coding:utf-8 -*-
from base import *
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB,GaussianNB
from sklearn.cross_validation import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
class EnsembleClassifier(object):
    def __init__(self, classifiers=None):
        self.classifiers = classifiers

    def fit(self, X, y):
        for classifier in self.classifiers:
            classifier.fit(X, y)

    def predict_proba(self, X):
        self.predictions_ = list()
        for classifier in self.classifiers:
            self.predictions_.append(classifier.predict_proba(X))
        return np.mean(self.predictions_, axis=0)

    def score(self,x,y):
        return 0
if __name__ == '__main__':
    X,Y = train_sets()
    gbc = GradientBoostingClassifier(n_estimators=100)
    lr = LogisticRegression()
    knn = neighbors.KNeighborsClassifier(n_neighbors=100)
    measure(EnsembleClassifier,{'classifiers':[gbc,lr,knn]},X,Y,'ensemble')

