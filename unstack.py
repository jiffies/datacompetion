#-*- coding:utf-8 -*-
import numpy as np
import pandas as pd
from object_bag import split_p

TRAINPATH = 'train-partion/%s.csv'
UNSTACKPATH = 'unstack-partion/%s.csv'
TESTUNSTACK = 'unstack-test/%s.csv'


def read_data(train=True):
    data = pd.read_csv('../train/log_train.csv',parse_dates=[1])
    test = pd.read_csv('../test/log_test.csv',parse_dates=[1])
    if train:
        index = data.enrollment_id.unique()
        m = data.groupby(['enrollment_id','object']).size()
    else:
        index = test.enrollment_id.unique()
        m = test.groupby(['enrollment_id','object']).size()
    return m,index
        


if __name__ == '__main__':
    m,index = read_data(train=False)
    split_point = split_p(len(index))
    split_point = [i-1 for i in split_point]
    split_point = index[split_point]
    last_point = 0
    for i in split_point:
        chunk = m.loc[last_point:i]
        chunk.unstack().fillna(0).to_csv(TESTUNSTACK % i)
        print "save unstack chunk %s to %s.\n" % (last_point,i)
        last_point = i+1
