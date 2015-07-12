#-*- coding:utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

log_train = pd.read_csv('../train/log_train.csv')
enroll_train = pd.read_csv('../train/enrollment_train.csv')
truth_train = pd.read_csv('../train/truth_train.csv',names=['enrollment_id','drop'])
log_train = pd.merge(log_train,enroll_train,on="enrollment_id",how="outer")
enroll_train = pd.merge(enroll_train,truth_train,on="enrollment_id",how="outer")

log_test = pd.read_csv('../test/log_test.csv')
enroll_test = pd.read_csv('../test/enrollment_test.csv')
truth_test = pd.read_csv('../test/truth_test.csv',names=['enrollment_id','drop'])
log_test = pd.merge(log_test,enroll_test,on="enrollment_id",how="outer")
enroll_test = pd.merge(enroll_test,truth_test,on="enrollment_id",how="outer")

drop_rate = enroll_train.groupby(['username','drop']).size().unstack(level=1).fillna(0)
drop_rate['rate'] = drop_rate[1]/(drop_rate[0]+drop_rate[1])
