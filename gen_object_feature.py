#-*- coding:utf-8 -*-
import numpy as np
import pandas as pd
from object_bag import split_p


TRAINPATH = 'train-partion/%s.csv'
TESTPATH = 'test-partion/%s.csv'
UNSTACKPATH = 'unstack-partion/%s.csv'
TESTUNSTACK = 'unstack-test/%s.csv'
TRAINOF = 'train-object-feature/%s.csv'
TESTOF = 'test-object-feature/%s.csv'

data = pd.read_csv('../train/log_train.csv',parse_dates=[1])
test = pd.read_csv('../test/log_test.csv',parse_dates=[1])

train = True
if train:
    index = data.enrollment_id.unique()
    split_point = split_p(len(index))
else:
    index = test.enrollment_id.unique()
    split_point = split_p(len(index))
u_point = [i-1 for i in split_point]
u_point = index[u_point]

if __name__ == "__main__":
    for i,(t,u) in enumerate(zip(split_point,u_point)):
        t = pd.read_csv(TESTPATH % t,index_col=0)
        u = pd.read_csv(TESTUNSTACK % u,index_col=0)
        result=t.add(u,fill_value=0)
        result.to_csv(TESTOF % i)
        print "generate feature file %s.\n" % i
