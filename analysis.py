#-*- coding:utf-8 -*-
import numpy as np
import scipy as sp
from datetime import datetime
import time

def timestr2stamp(timestr):
    fmt = "%Y-%m-%dT%H:%M:%S"
    return time.mktime(datetime.strptime(timestr,fmt).timetuple())


data = np.genfromtxt('../train/log_train.csv',delimiter=",",names=True,usecols=(0,1),converters={1:timestr2stamp})

print data


