#-*- coding:utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from functools import partial
from sklearn import neighbors
from sklearn.cross_validation import KFold
from sklearn import preprocessing 
from matplotlib import pylab
import time
import datetime
import multiprocessing

def avgtimeperday(df):
    t=df['time']
    return t.max()-t.min()
def most_interest_object(group):
    return np.sum(group['object'].value_counts()>10)

def object_num(df):
    return np.size(df['object'].value_counts())

def distance(df):
    return df['time'].max()-df['time'].min()

class Extractor(object):
    def __init__(self,enroll,log,truth=None):
        log['date'] = log['time'].dt.date
        enroll_log = pd.merge(log,enroll,on="enrollment_id",how="outer")
        self.log = log
        self.enroll_log = enroll_log
        self.label= truth
        self.task_queue = multiprocessing.JoinableQueue()
        self.result_queue = multiprocessing.Queue()
        self.tasks = self.extractor_names()
        self.pool_size = 3 #multiprocessing.cpu_count()
        self.features = []
        if self.label is not None:
            self.features.append(self.label)



    def f_visit(self):
        data = self.log
        g_enrollment_id = data.groupby('enrollment_id')
        visit = g_enrollment_id.size()
        v=pd.DataFrame({'enrollment_id':visit.index,'visit_time':visit.values})
        return v

    def f_lv(self):
        data = self.log
        g_enrollment_id = data.groupby('enrollment_id')
        lv = g_enrollment_id
        lv =lv.aggregate({'time':np.max})
        max = data['time'].max()
        lv=max-lv['time']
        lv=lv.dt.days
        lv = pd.DataFrame({'enrollment_id':lv.index,'last_visit':lv.values})
        return lv

    def f_time_per_visit(self):
        data = self.log
        g_enrollment_id = data.groupby('enrollment_id')
        visit = g_enrollment_id.size()
        day_span = g_enrollment_id.apply(distance)
        day_span = day_span.dt.days/visit
        day_span=day_span.reset_index().rename_axis({0:'time_per_visit'},axis=1)
        return day_span

    def f_last_month(self):
        data = self.log
        last_mouth = data[data['time']>'2014-07-01T17:31:15']
        last_mouth_visit=last_mouth.groupby('enrollment_id').size()
        lmv=pd.DataFrame({'enrollment_id':last_mouth_visit.index,'last_mouth_visit':last_mouth_visit.values})
        return lmv

    def f_event(self):
        data = self.log
        event = data.groupby(['enrollment_id','event']).size().unstack().fillna(0)
        event.reset_index(inplace=True)
        return event[['enrollment_id','discussion','video','problem']]

    def f_source(self):
        data = self.log
        source = data.groupby(['enrollment_id','source']).size().unstack().fillna(0)
        source.reset_index(inplace=True)
        return source

    def f_mic(self):
        data = self.log
        g_enrollment_id = data.groupby('enrollment_id')
        g = g_enrollment_id
        mic=g.apply(most_interest_object) #most_interest_class,object>10
        mic=pd.DataFrame({'enrollment_id':mic.index,'mic':mic.values})
        return mic

    def f_object_num(self):
        data = self.log
        g_enrollment_id = data.groupby('enrollment_id')
        g = g_enrollment_id
        c_num = g.apply(object_num)
        c_num=pd.DataFrame({'enrollment_id':c_num.index,'object_num':c_num.values})
        return c_num

    def f_avg_second_per_day(self):
        data = self.log
        avg_second_per_day = data.groupby(['enrollment_id','date']).apply(avgtimeperday)
        avg_second_per_day = avg_second_per_day.dt.seconds
        avg_second_per_day = avg_second_per_day.groupby(axis=0,level=0).mean()
        avg_second_per_day=pd.DataFrame({'enrollment_id':avg_second_per_day.index,'avg_second_per_day':avg_second_per_day.values})
        return avg_second_per_day

    def f_user_visit(self):
        log = self.enroll_log
        user_visit = log.groupby('username').size().reset_index().rename_axis({0:'user_visit'},axis=1)
        user_visit = pd.merge(log,user_visit,on='username',how='outer')
        user_visit = user_visit[['enrollment_id','user_visit']]
        user_visit = user_visit.groupby('enrollment_id').mean().reset_index()
        return user_visit

    def f_user_last_month(self):
        log = self.enroll_log
        user_last_mouth = log[log['time']>(log['time'].max()-datetime.timedelta(30))]
        user_last_mouth_visit = user_last_mouth.groupby('username').size().reset_index().rename_axis({0:'user_last_mouth_visit'},axis=1)
        user_last_mouth_visit = pd.merge(log,user_last_mouth_visit,on='username',how='outer')
        user_last_mouth_visit = user_last_mouth_visit[['enrollment_id','user_last_mouth_visit']]
        user_last_mouth_visit = user_last_mouth_visit.groupby('enrollment_id').mean().reset_index()
        return user_last_mouth_visit


    def f_user_lv(self):
        log = self.enroll_log
        data = self.log
        user_lv = log.groupby('username')
        user_lv =user_lv.aggregate({'time':np.max})
        max = data['time'].max()
        user_lv=max-user_lv['time']
        user_lv=user_lv.dt.days
        user_lv = pd.DataFrame({'username':user_lv.index,'user_lv':user_lv.values})
        user_lv = pd.merge(log,user_lv,on='username',how='outer')
        user_lv = user_lv[['enrollment_id','user_lv']]
        user_lv = user_lv.groupby('enrollment_id').mean().reset_index()
        return user_lv

    def f_user_event(self):
        log = self.enroll_log
        user_event = log.groupby(['username','event']).size().unstack().fillna(0)
        user_event.reset_index(inplace=True)
        user_event = user_event[['username','discussion','video','problem','access','wiki','nagivate','page_close']]
        user_event = pd.merge(log,user_event,on='username',how='outer')
        user_event = user_event[['enrollment_id','discussion','video','problem','access','wiki','nagivate','page_close']]
        user_event = user_event.groupby('enrollment_id').mean().reset_index()
        return user_event

    def extractor_names(self):
        names = []
        for fun in dir(self):
            if fun.startswith('f_'):
                names.append(fun)
        return names


    def worker(self):
        p = multiprocessing.current_process()
        print "%s pid: %d" % (p.name,p.pid)
        while True:
            task = self.task_queue.get()
            if task is None:
                print "%s拿到毒丸，结束进程." % p.name
                self.task_queue.task_done()
                break
            print "%s拿到任务: %s..." % (p.name,task)
            result = getattr(self,task)()
            self.task_queue.task_done()
            self.result_queue.put(result)
            print "%s完成任务%s..." % (p.name,task)
        return

    def start(self):
        jobs = []
        for i in xrange(self.pool_size):
            p = multiprocessing.Process(target=self.worker,name="%d进程" % i)
            jobs.append(p)
            p.start()
        for task_name in self.tasks:
            print "放入任务",task_name
            self.task_queue.put(task_name)
        for i in xrange(self.pool_size):
            self.task_queue.put(None)
        self.task_queue.join()
        print 'task finish'
        for task_name in self.tasks:
            result = self.result_queue.get()
            self.features.append(result)
        features = reduce(partial(pd.merge,on='enrollment_id',how='outer'),self.features)
        features = features.fillna(0)
        print features

        for p in jobs:
            p.join()
        print "特征提取结束."
        return features

def worker(extractor,task_queue,result_queue):
    p = multiprocessing.current_process()
    print "pid:",p.pid
    task = task_queue.get()
    print "拿到任务: %s..." % task
    result = getattr(extractor,task)()
    task_queue.task_done()
    result_queue.put(result)
    result_queue.close()
    print "close"
    result_queue.join_thread()
    print "完成任务..."
    return






def extract_features(data,log,label=None):
    data['date'] = data['time'].dt.date
    features = []
    user_features = []
    g_enrollment_id = data.groupby('enrollment_id')
    if label is not None:
        features.append(label)
    visit = g_enrollment_id.size()
    v=pd.DataFrame({'enrollment_id':visit.index,'visit_time':visit.values})
    features.append(v)

    course_visit = log.groupby(['course_id','enrollment_id']).size().mean(level=0).reset_index().rename_axis({0:'course_visit'},axis=1)
    course_visit = pd.merge(log,course_visit,on='course_id',how='outer')
    course_visit = course_visit[['enrollment_id','course_visit']]
    course_visit = course_visit.groupby('enrollment_id').mean().reset_index()
    features.append(course_visit)


    course_lv = log.groupby('course_id')
    course_lv =course_lv.aggregate({'time':np.max})
    max = data['time'].max()
    course_lv=max-course_lv['time']
    course_lv=course_lv.dt.days
    course_lv = pd.DataFrame({'course_id':course_lv.index,'course_lv':course_lv.values})
    course_lv = pd.merge(log,course_lv,on='course_id',how='outer')
    course_lv = course_lv[['enrollment_id','course_lv']]
    course_lv = course_lv.groupby('enrollment_id').mean().reset_index()
    features.append(course_lv)

    user_visit = log.groupby('username').size().reset_index().rename_axis({0:'user_visit'},axis=1)
    user_visit = pd.merge(log,user_visit,on='username',how='outer')
    user_visit = user_visit[['enrollment_id','user_visit']]
    user_visit = user_visit.groupby('enrollment_id').mean().reset_index()
    features.append(user_visit)

    user_last_mouth = log[log['time']>(log['time'].max()-datetime.timedelta(30))]
    user_last_mouth_visit = user_last_mouth.groupby('username').size().reset_index().rename_axis({0:'user_last_mouth_visit'},axis=1)
    user_last_mouth_visit = pd.merge(log,user_last_mouth_visit,on='username',how='outer')
    user_last_mouth_visit = user_last_mouth_visit[['enrollment_id','user_last_mouth_visit']]
    user_last_mouth_visit = user_last_mouth_visit.groupby('enrollment_id').mean().reset_index()
    features.append(user_last_mouth_visit)


    user_lv = log.groupby('username')
    user_lv =user_lv.aggregate({'time':np.max})
    max = data['time'].max()
    user_lv=max-user_lv['time']
    user_lv=user_lv.dt.days
    user_lv = pd.DataFrame({'username':user_lv.index,'user_lv':user_lv.values})
    user_lv = pd.merge(log,user_lv,on='username',how='outer')
    user_lv = user_lv[['enrollment_id','user_lv']]
    user_lv = user_lv.groupby('enrollment_id').mean().reset_index()
    features.append(user_lv)

    user_event = log.groupby(['username','event']).size().unstack().fillna(0)
    user_event.reset_index(inplace=True)
    user_event = user_event[['username','discussion','video','problem','access','wiki','nagivate','page_close']]
    user_event = pd.merge(log,user_event,on='username',how='outer')
    user_event = user_event[['enrollment_id','discussion','video','problem','access','wiki','nagivate','page_close']]
    user_event = user_event.groupby('enrollment_id').mean().reset_index()
    features.append(user_event)


    day_span = g_enrollment_id.apply(distance)
    day_span = day_span.dt.days/visit
    day_span=day_span.reset_index().rename_axis({0:'day_span'},axis=1)
    features.append(day_span)

    last_mouth = data[data['time']>'2014-07-01T17:31:15']
    last_mouth_visit=last_mouth.groupby('enrollment_id').size()
    lmv=pd.DataFrame({'enrollment_id':last_mouth_visit.index,'last_mouth_visit':last_mouth_visit.values})
    features.append(lmv)

    ##last_week = data[data['time']>'2014-07-24T17:31:15']
    ##last_week_visit=last_week.groupby('enrollment_id').size()
    ##lwv=pd.DataFrame({'enrollment_id':last_week_visit.index,'last_week_visit':last_week_visit.values})
    ##features.append(lwv)

    lv = g_enrollment_id
    lv =lv.aggregate({'time':np.max})
    max = data['time'].max()
    lv=max-lv['time']
    lv=lv.dt.days
    lv = pd.DataFrame({'enrollment_id':lv.index,'last_visit':lv.values})
    features.append(lv)

    event = data.groupby(['enrollment_id','event']).size().unstack().fillna(0)
    event.reset_index(inplace=True)
    features.append(event[['enrollment_id','discussion','video','problem']])

    source = data.groupby(['enrollment_id','source']).size().unstack().fillna(0)
    source.reset_index(inplace=True)
    features.append(source)

    g = g_enrollment_id
    mic=g.apply(most_interest_object) #most_interest_class,object>10
    mic=pd.DataFrame({'enrollment_id':mic.index,'mic':mic.values})
    features.append(mic)

    c_num = g.apply(object_num)
    c_num=pd.DataFrame({'enrollment_id':c_num.index,'c_num':c_num.values})
    features.append(c_num)

    avg_second_per_day = data.groupby(['enrollment_id','date']).apply(avgtimeperday)
    avg_second_per_day = avg_second_per_day.dt.seconds
    avg_second_per_day = avg_second_per_day.groupby(axis=0,level=0).mean()
    avg_second_per_day=pd.DataFrame({'enrollment_id':avg_second_per_day.index,'avg_second_per_day':avg_second_per_day.values})
    features.append(avg_second_per_day)

    
    result = reduce(partial(pd.merge,on='enrollment_id',how='outer'),features)
    result = result.fillna(0)
    return result


def measure(clf_class,parameters,x,y,name,data_size=None, plot=False):
    print "开始测试..."
    start_time = time.time()
    if data_size is None:
        X = x
        Y = y
    else:
        X = x[:data_size]
        Y = y[:data_size]

    scores = []
    train_errors = []
    test_errors = []
    roc_auc = []

    precisions, recalls, thresholds = [], [], []
    cv = KFold(n=len(X),n_folds=5,shuffle=True)

    for train,test in cv:
        x_train,y_train = X[train],Y[train]
        x_test,y_test = X[test],Y[test]
        clf = clf_class(**parameters)
        #clf = neighbors.KNeighborsClassifier(n_neighbors=100)
        clf.fit(x_train,y_train)

        train_score = clf.score(x_train, y_train)
        test_score = clf.score(x_test, y_test)

        train_errors.append(1 - train_score)
        test_errors.append(1 - test_score)
        print "train_error =%f\ttest_error =%f\n" % (1-train_score,1-test_score)

        scores.append(test_score)
        proba = clf.predict_proba(x_test)
        roc_auc.append(roc_auc_score(y_test,proba[:,1]))

    print "测试完毕"
    print "%s use time %s" % (name,time.time()-start_time)
    print "Score Mean = %.5f\tStddev = %.5f" % (np.mean(scores),np.std(scores))
    print "auc Mean = %.5f\tauc Stddev = %.5f" % (np.mean(roc_auc),np.std(roc_auc))
    return np.mean(train_errors), np.mean(test_errors)

def train_sets(reproduct=False):
    print "读取训练数据..."
    if reproduct:
        data = pd.read_csv('../train/log_train.csv',parse_dates=[1])
        enroll_train = pd.read_csv('../train/enrollment_train.csv')
        truth_train = pd.read_csv('../train/truth_train.csv',names=['enrollment_id','drop'])
        log_train = pd.merge(data,enroll_train,on="enrollment_id",how="outer")
        enroll_train = pd.merge(enroll_train,truth_train,on="enrollment_id",how="outer")
        result = extract_features(data,log_train,label=truth_train)
        result.to_csv('features.csv')
    else:
        result = pd.read_csv('features.csv',index_col=0)
        #X = np.genfromtxt("features.csv")
        #Y = np.genfromtxt("labels.csv")
    print result
    X = result.ix[:,2:].values
    Y = result['drop'].real
    #X = preprocessing.normalize(X,axis=0)
#X = visit.real.reshape(len(visit),1)
    return X,Y




def test_sets():

    print "读取测试数据..."
    test = pd.read_csv('../test/log_test.csv',parse_dates=[1])

    enroll_test = pd.read_csv('../test/enrollment_test.csv')
    log_test = pd.merge(test,enroll_test,on="enrollment_id",how="outer")

    visit_test = test.groupby('enrollment_id').size()
    test_features = extract_features(test,log_test)
    test_features.to_csv('test-features.csv')
    print test_features
    X_predict = test_features.ix[:,1:].values
    #X_predict = preprocessing.normalize(X_predict,axis=0)
    return X_predict,visit_test

def gen_submission(clf_class,parameters,x,y,name):
    X,Y = x,y
    clf = clf_class(**parameters)
    print "在整个训练集上训练模型..."
    clf.fit(X,Y)
    print "开始预测..."
    X_predict,visit_test = test_sets()
    Y_predict = clf.predict_proba(X_predict)
    submission = np.vstack((visit_test.index.values,Y_predict[:,1])).transpose()

    import datetime
    np.savetxt('%s-%s.csv' % (name,datetime.datetime.now()),submission,fmt="%d,%f")
    print "结果写入result文件"


def plot_bias_variance(data_sizes, train_errors, test_errors, name, title):
    pylab.figure(num=None, figsize=(6, 5))
    pylab.ylim([0.0, 1.0])
    pylab.xlabel('Data set size')
    pylab.ylabel('Error')
    pylab.title("Bias-Variance for '%s'" % name)
    pylab.plot(
        data_sizes, test_errors, "--", data_sizes, train_errors, "b-", lw=1)
    pylab.legend(["train error", "test error"], loc="upper right")
    pylab.grid(True, linestyle='-', color='0.75')
    pylab.savefig("bv_" + name.replace(" ", "_") + ".png", bbox_inches="tight")

def bias_variance_analysis(clf_class, parameters,x,y, name):
    data_sizes = np.arange(1000, 10000, 1000)
    X,Y = x,y

    train_errors = []
    test_errors = []

    for data_size in data_sizes:
        train_error, test_error = measure(
            clf_class, parameters, X, Y, name, data_size=data_size)
        train_errors.append(train_error)
        test_errors.append(test_error)

    plot_bias_variance(data_sizes, train_errors, test_errors, name, "Bias-Variance for '%s'" % name)
    
if __name__ == '__main__':

    #log_train  = pd.read_csv('../train/log_train.csv',parse_dates=[1])
    #enroll_train = pd.read_csv('../train/enrollment_train.csv')
    #truth_train = pd.read_csv('../train/truth_train.csv',names=['enrollment_id','drop'])
    #start = time.time()
    #enroll_train = pd.merge(log_train,enroll_train,on="enrollment_id",how="outer")
    #f = extract_features(log_train,enroll_train,truth_train)
    #print f
    #e = Extractor(enroll_train,log_train,truth_train)
    #e.start()
    #end = time.time()
    #print "use time %f " % (end-start)
    #task_queue = multiprocessing.JoinableQueue()
    #result_queue = multiprocessing.Queue()
    #p = multiprocessing.Process(target=worker,args=(e,task_queue,result_queue))
    #p.start()
    #task_queue.put('f_visit')
    #print "放入任务"
    #task_queue.join()
    #print 'task finish'
    #result = result_queue.get()
    #print "get"

    #print result
    #p.join()
    #print "over"
    
    X,Y = train_sets()
    #bias_variance_analysis(neighbors.KNeighborsClassifier,{'n_neighbors':100},X,Y,'knn')
    measure(neighbors.KNeighborsClassifier,{'n_neighbors':100},X,Y,'knn')
    #gen_submission(neighbors.KNeighborsClassifier,{'n_neighbors':100},X,Y,'knn')

