#-*- coding:utf-8 -*-
import numpy as np
import scipy as sp
import os.path
CURRENT = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.split(CURRENT)[0]

data = np.genfromtxt(os.path.join(ROOT,'sampleSubmission.csv'),delimiter=",")
y = data[:,1]
data[:,1]+=np.random.random(y.shape)

np.savetxt("random_submission.csv",data,fmt="%d,%f")
