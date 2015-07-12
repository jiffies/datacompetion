#-*- coding:utf-8 -*-
from gen_object_feature import *
from sklearn.linear_model import SGDClassifier
import datetime
OBJSUB = 'object-submission/%s.csv'

truth_train = pd.read_csv('../train/truth_train.csv',index_col=0,names=['drop'])
def yield_xy(train=True):
    last_point = 0
    for index,i in enumerate(split_point):
        if train:
            Y = truth_train[last_point:i].values.ravel()
            last_point = i
            X = pd.read_csv(TRAINOF % index,index_col=0).values
            yield X,Y
        else:
            chunk = pd.read_csv(TESTOF % index,index_col=0)
            X = chunk.values
            label = chunk.index.values
            yield X,label

if __name__ == '__main__':
    clf = SGDClassifier(loss="log")
    for i,(X,Y) in enumerate(yield_xy()):
        if i==1:
            x,y = X,Y
            continue
            print "score %f\n" % clf.score(X,Y)
            break
        clf.partial_fit(X,Y,classes=[0,1])
        print "partial train round %d\n" % i
    print "score %f\n" % clf.score(x,y)
    
    #import pickle
    #from sklearn.externals import joblib
    #joblib.dump(clf,'object-clf.pkl')
 
    ##clf = joblib.load'object-clf.pkl')
    #for i,(X,label) in enumerate(yield_xy(train=False)):
        #proba = clf.predict_proba(X)
        #submission = np.vstack((label,proba[:,1])).transpose()
        #np.savetxt(OBJSUB % i,submission,fmt="%d,%f")
        #print "save submission %d.\n" % i

    #with open('result-object-%s.csv' % datetime.datetime.now(),'a') as result:
        #for a in [OBJSUB % i for i in range(10)]:
            #chunk = file(a).read()
            #result.write(chunk)

    


    

