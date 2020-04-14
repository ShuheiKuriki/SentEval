import numpy as np
import time
from rvsml.NNClassifier import *
from rvsml.RVSML_OT_Learning import *
import logging
from array import array

def EvaluateRVSML_dtw(classnum, dim, trainset, trainsetnum, testset, testsetdata, testsetdatanum, testsetlabel, testsetnum, logname, logfile):
    #rankdim = 58
    CVAL = 1

    logger = logging.getLogger(logname)

    logger.setLevel(10)

    sh = logging.StreamHandler()
    logger.addHandler(sh)

    logging.basicConfig(filename=logfile, format="%(message)s", filemode='w')
    # addpath('E:/BING/ActionRecognition/FrameWideFeatures/libsvm-3.20/matlab')

    delta = 1
    lambda1 = 50
    lambda2 = 0.1
    max_iters = 10
    err_limit = 10**(-2)

    class Options:
        def __init__(self, max_iters, err_limit, lambda1, lambda2, delta):
            self.max_iters = max_iters
            self.err_limit = err_limit
            self.lambda1 = lambda1
            self.lambda2 = lambda2
            self.delta = delta

    options = Options(max_iters,err_limit,lambda1,lambda2,delta)


    # testsetlabel = testsetlabel.T

    # for name in matlist:
    #     exec("%s = %s[0]" % (name, name))

    # testsetdatanum = testsetdatanum[0]

    trainset_m = trainset

    # shape=trainset_m[0].shape

    # for c in range(classnum):
    #     for m in range(trainsetnum[c]):
    #         trainset_m[c][0][m] = trainset[c][0][m] - traindatamean

    # testsetdata_m = testsetdata

    # for m in range(testsetdatanum):
    #     testsetdata_m[m] = testsetdata[m] - traindatamean

    ## RVSML-DTW

    
    logger.info("data load done")

    logger.info("DTW start")

    templatenum = 4
    lambda0 = 0.0005
    tic = time.time()
    L = RVSML_OT_Learning_dtw(trainset_m, trainsetnum, dim,templatenum,lambda0,options,logname)
    RVSML_dtw_time = time.time() - tic

    logger.info("DTW learning done")
    ## classification with the learned metric
    traindownset = [0]*classnum
    testdownsetdata = [0]*testsetdatanum
    for j in range(classnum):
        traindownset[j] = [0]*trainsetnum[j]
        for m in range(trainsetnum[j]):
            traindownset[j][m] = np.dot(trainset[j][m] ,L)

    for j in range(testsetdatanum):
        testdownsetdata[j] = np.dot(testsetdata[j], L)

    RVSML_dtw_macro, RVSML_dtw_micro, RVSML_dtw_acc, dtw_knn_time, dtw_knn_average_time = NNClassifier_dtw(classnum,traindownset,trainsetnum,testdownsetdata,testsetdatanum,testsetlabel,options,logname)
    RVSML_dtw_acc_1 = RVSML_dtw_acc[0]

    logger.info('Training time of RVSML instantiated by DTW is {:.4f} \n'.format(RVSML_dtw_time))
    logger.info('Classification using 1 nearest neighbor classifier with DTW distance:\n')
    logger.info('MAP macro is {:.4f}, micro is {:.4f} \n'.format(RVSML_dtw_macro, RVSML_dtw_micro))
    logger.info('dtw_knn_time is {:.4f} \n'.format(dtw_knn_time))
    logger.info('dtw_knn_average_time is {:.4f} \n'.format(dtw_knn_average_time))

    for acc in RVSML_dtw_acc:
        logger.info('Accuracy is {:.4f} \n'.format(acc))

    return