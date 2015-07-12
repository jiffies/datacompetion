#-*- coding:utf-8 -*-
import numpy as np
import pandas as pd

TRAINPATH = 'train-partion/%s.csv'
TESTPATH = 'test-partion/%s.csv'

def gen_table(train=True):
    data = pd.read_csv('../train/log_train.csv',parse_dates=[1])
    test = pd.read_csv('../test/log_test.csv',parse_dates=[1])
    bag = np.union1d(data.object.unique(),test.object.unique())
    if train:
        index = data.enrollment_id.unique()
    else:
        index = test.enrollment_id.unique()
    table = pd.DataFrame(0,index=index,columns=bag,dtype='int16')
    return table,index

def split_p(length,partion=10):
    split_point = [i*(length/partion) for i in range(1,partion)]
    split_point.append(length)
    return split_point


def split_table(df,split_point,path=TRAINPATH):
    last_point = 0
    for i in split_point:
        print "save data %s to %s.\n" % (last_point,i)
        df[last_point:i].to_csv(path % i)
        last_point = i

if __name__ == '__main__':
    table,index = gen_table(train=False)
    split_point= split_p(len(index))
    print split_point
    split_table(table,split_point,path=TESTPATH)




