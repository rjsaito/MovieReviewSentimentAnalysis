from __future__ import division, print_function
import sys
import numpy as np
from numpy import array, asarray, bincount, count_nonzero, genfromtxt, mean, median, percentile, unique, argsort, zeros_like, sort, sum, percentile, all, arange, argmax, linspace, amin, zeros
import random as rnd
from numpy.random import choice
import pandas as pd
import math
from math import log, exp
import time
from os import listdir
from os.path import isfile, join
import sklearn
import sklearn.feature_extraction
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.manifold import locally_linear_embedding, MDS
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA, KernelPCA
from sklearn.linear_model import Ridge, LogisticRegression, SGDClassifier
from sklearn.kernel_ridge import KernelRidge

def read_txt(filename):
    assert isfile(filename)
    f = open(filename)
    data = f.read()
    return data

def read_file(filename):
    assert isfile(filename)
    print("Loading", filename)
    data = np.genfromtxt(filename, delimiter=",", skip_header=0)
    print("Successfully Loaded", filename)
    print("")
    return data

### Data Set 1: Sentiment Analysis (2-class Classification)
# Read files
path_neg = "C:/Users/rjsai/Dropbox/UMN Courses/CSCI 5525/project/review_polarity/txt_sentoken/neg/"
path_pos = "C:/Users/rjsai/Dropbox/UMN Courses/CSCI 5525/project/review_polarity/txt_sentoken/pos/"

neg_reviews = [read_txt(join(path_neg , f)) for f in listdir(path_neg)]
pos_reviews = [read_txt(join(path_pos , f)) for f in listdir(path_pos)]
all_reviews = neg_reviews + pos_reviews

# Create Term Document Matrices
#TDM w/ each word as columns (removing any words freq < min_df
#and freq > 80% of reviews
vc = sklearn.feature_extraction.text.CountVectorizer(min_df=5, max_df=0.8)
Xorig = vc.fit_transform(all_reviews).toarray()

#12/10/2016: New X_ppos data
#file = "C:/Users/rjsai/Dropbox/UMN Courses/CSCI 5525/project/X1_ppos.txt"
#Xppos = np.genfromtxt(file, delimiter=" ", skip_header=0) #delim_whitespace=True


#TDM w/ 2-gram
vc_2gram = sklearn.feature_extraction.text.CountVectorizer(min_df=5, max_df=0.8, ngram_range=(1, 2))
X2gram = vc_2gram.fit_transform(all_reviews).toarray()

#TDM w/ TF-IDF (instead of counts)
vc_tfidf = sklearn.feature_extraction.text.TfidfVectorizer(min_df=5, max_df=0.8)
Xtfidf = vc_tfidf.fit_transform(all_reviews).toarray()

# Polarity
y = np.array([-1] * 1000 + [1] * 1000)

# Dimension Reduction
##Principal Components
pca = PCA(n_components=2000) 
pca.fit(Xorig)
Xpc = pca.fit_transform(Xorig)
Xpc_2g = pca.fit_transform(X2gram)
Xpc_tfidf = pca.fit_transform(Xtfidf)


##Kernel Principal Components
kpca = KernelPCA(kernel="rbf", fit_inverse_transform=True, gamma=10)
Xkpc = kpca.fit_transform(Xorig)
Xkpc_2g = kpca.fit_transform(X2gram)
Xkpc_tfidf = kpca.fit_transform(Xtfidf)


# Naive Bayes with cross validation 
def NBCV(X, y, K):

    gnb = GaussianNB()
    K = int(K)
    n = y.size

    #initialize
    trainerror = []
    testerror = []
    train_cm = np.array([[0,0],[0,0]])
    test_cm = np.array([[0,0],[0,0]]) 

    index = np.arange(n)
    np.random.shuffle(index)
    index = index[np.argsort(y[index], kind='mergesort')]
    
    for fold in range(0, K):
        #print('Working on fold', fold, '...')
        train_index = np.remainder(index, K) != fold
        test_index = np.remainder(index, K) == fold
        X_train = X[train_index, :]
        y_train = y[train_index]
        X_test = X[test_index, :]
        y_test = y[test_index]

        yhat_train = gnb.fit(X_train, y_train).predict(X_train)
        yhat_test = gnb.fit(X_train, y_train).predict(X_test)

        trainerror.append(np.mean(y_train != yhat_train))
        testerror.append(np.mean(y_test != yhat_test))
        avg_trainerror = np.mean(trainerror)
        avg_testerror = np.mean(testerror)

        train_cm += confusion_matrix(y_train, yhat_train)
        test_cm += confusion_matrix(y_test, yhat_test)

    with open("nb_error.dat", "w") as f:
        for k in range(K):
            print(trainerror[k], end=", ", file=f)
            print(testerror[k], end=", ", file=f)
        print(file=f)

    print('')
    print('Naive Bayes')
    print('Cross Validation (', K, '-Fold) Complete', sep='')
    print('')
    print('Training Error:', ['%.4f' % i for i in trainerror])
    print('Training SD:', ['%.4f' % np.std(trainerror)])
    print('Testing Error:', ['%.4f' % i for i in testerror])
    print('Testing SD:', ['%.4f' %np.std(testerror)])
    print('Average Training Error:', round(avg_trainerror, 4))
    print('Average Testing Error:', round(avg_testerror, 4))
    print('')
    print('Training Confusion Matrix')
    print(train_cm)
    print('')
    print('Testing Confusion Matrix')
    print(test_cm)
    print('')
    print('')
    

# Logistic Regression with cross validation 
def LRCV(X, y, K, penalty='l2', C = 1):

    lr = LogisticRegression(penalty = penalty, C = C)
    K = int(K)
    n = y.size
    y_vals = unique(y)
    #initialize
    trainerror = []
    testerror = []
    train_cm = np.array([[0,0],[0,0]])
    test_cm = np.array([[0,0],[0,0]]) 

    index = np.arange(n)
    np.random.shuffle(index)
    index = index[np.argsort(y[index], kind='mergesort')]
    
    for fold in range(0, K):
        #print('Working on fold', fold, '...')
        train_index = np.remainder(index, K) != fold
        test_index = np.remainder(index, K) == fold
        X_train = X[train_index, :]
        y_train = y[train_index]
        X_test = X[test_index, :]
        y_test = y[test_index]

        yhat_train = lr.fit(X_train, y_train).predict(X_train)
        yhat_test = lr.fit(X_train, y_train).predict(X_test)
        yhat_train = [y_vals[0] if s < 0 else y_vals[1] for s in yhat_train]
        yhat_test = [y_vals[0] if s < 0 else y_vals[1] for s in yhat_test]

        trainerror.append(np.mean(y_train != yhat_train))
        testerror.append(np.mean(y_test != yhat_test))
        avg_trainerror = np.mean(trainerror)
        avg_testerror = np.mean(testerror)    

        train_cm += confusion_matrix(y_train, yhat_train)
        test_cm += confusion_matrix(y_test, yhat_test)
        
    with open("lr_error.dat", "w") as f:
        for k in range(K):
            print(trainerror[k], end=", ", file=f)
            print(testerror[k], end=", ", file=f)
        print(file=f)

    print('')
    print('Logistic Regression: penalty = ', penalty, ', C = ', C)
    print('Cross Validation (', K, '-Fold) Complete', sep='')
    print('')
    print('Training Error:', ['%.4f' % i for i in trainerror])
    print('Training SD:', ['%.4f' % np.std(trainerror)])
    print('Testing Error:', ['%.4f' % i for i in testerror])
    print('Testing SD:', ['%.4f' %np.std(testerror)])
    print('Average Training Error:', round(avg_trainerror, 4))
    print('Average Testing Error:', round(avg_testerror, 4))
    print('')
    print('Training Confusion Matrix')
    print(train_cm)
    print('')
    print('Testing Confusion Matrix')
    print(test_cm)
    print('')
    print('')


# Elastic Net with cross validation 
def EnetCV(X, y, K, l1_ratio = 0.15, penalty = 'elasticnet'):

    #alpha = 1.0, l1_ratio = 0.5,
    enet = SGDClassifier(l1_ratio = l1_ratio, penalty = penalty, loss = "log")
    #enet = ElasticNet(alpha = alpha, l1_ratio = l1_ratio)
    
    K = int(K)
    n = y.size
    y_vals = unique(y)
    #initialize
    trainerror = []
    testerror = []
    train_cm = np.array([[0,0],[0,0]])
    test_cm = np.array([[0,0],[0,0]])
    
    index = np.arange(n)
    np.random.shuffle(index)
    index = index[np.argsort(y[index], kind='mergesort')]
    
    for fold in range(0, K):
        #print('Working on fold', fold, '...')
        train_index = np.remainder(index, K) != fold
        test_index = np.remainder(index, K) == fold
        X_train = X[train_index, :]
        y_train = y[train_index]
        X_test = X[test_index, :]
        y_test = y[test_index]

        yhat_train = enet.fit(X_train, y_train).predict(X_train)
        yhat_test = enet.fit(X_train, y_train).predict(X_test)
        yhat_train = [y_vals[0] if s < 0 else y_vals[1] for s in yhat_train]
        yhat_test = [y_vals[0] if s < 0 else y_vals[1] for s in yhat_test]

        trainerror.append(np.mean(y_train != yhat_train))
        testerror.append(np.mean(y_test != yhat_test))
        avg_trainerror = np.mean(trainerror)
        avg_testerror = np.mean(testerror)    

        train_cm += confusion_matrix(y_train, yhat_train)
        test_cm += confusion_matrix(y_test, yhat_test)
        
    with open("enet_error.dat", "w") as f:
        for k in range(K):
            print(trainerror[k], end=", ", file=f)
            print(testerror[k], end=", ", file=f)
        print(file=f)

    print('')
    print('Elastic Net: l1_ratio = ', l1_ratio)
    print('Cross Validation (', K, '-Fold) Complete', sep='')
    print('')
    print('Training Error:', ['%.4f' % i for i in trainerror])
    print('Training SD:', ['%.4f' % np.std(trainerror)])
    print('Testing Error:', ['%.4f' % i for i in testerror])
    print('Testing SD:', ['%.4f' %np.std(testerror)])
    print('Average Training Error:', round(avg_trainerror, 4))
    print('Average Testing Error:', round(avg_testerror, 4))
    print('')
    print('Training Confusion Matrix')
    print(train_cm)
    print('')
    print('Testing Confusion Matrix')
    print(test_cm)
    print('')
    print('')

#######################################################################

print("X (Original)")
NBCV(Xorig, y, 5)
print("X (Original, 2-gram)")
NBCV(X2gram, y, 5)
print("X (Original, TFIDF)")
NBCV(Xtfidf, y, 5)
print("X (Kernel Principal Components)")
NBCV(Xkpc, y, 5)
print("X (Kernel Principal Components, 2-gram)")
NBCV(Xkpc_2g, y, 5)
print("X (Kernel Principal Components, TFIDF)")
NBCV(Xkpc_tfidf, y, 5)

#print("X (Original)")
#LRCV(Xorig, y, 5)
#print("X (Original)")
#LRCV(Xorig, y, 5, penalty = 'l1')
#print("X (Original)")
#LRCV(Xorig, y, 5, C = 4)
#print("X (Original)")
#LRCV(Xorig, y, 5, penalty = 'l1', C = 4)
#print("X (Original, 2-gram)")
#LRCV(X2gram, y, 5)
#print("X (Original, 2-gram)")
#LRCV(X2gram, y, 5, penalty = 'l1')
#print("X (Original, TFIDF)")
#LRCV(Xtfidf, y, 5)
#print("X (Original, TFIDF)")
#LRCV(Xtfidf, y, 5, penalty = 'l1')


print("X (Principal Components)")
LRCV(Xpc, y, 5, penalty = 'l1')
print("X (KPrincipal Components, 2-gram)")
LRCV(Xpc_2g, y, 5, penalty = 'l1')
print("X (Principal Components, TFIDF)")
LRCV(Xpc_tfidf, y, 5, penalty = 'l1')

print("X (Kernel Principal Components)")
LRCV(Xkpc, y, 5)
print("X (Kernel Principal Components)")
LRCV(Xkpc, y, 5, penalty = 'l1')
print("X (Kernel Principal Components)")
LRCV(Xkpc, y, 5, C = 4)
print("X (Kernel Principal Components)")
LRCV(Xkpc, y, 5, penalty = 'l1', C = 4)
print("X (Kernel Principal Components, 2-gram)")
LRCV(Xkpc_2g, y, 5)
print("X (Kernel Principal Components, 2-gram)")
LRCV(Xkpc_2g, y, 5, penalty = 'l1')
print("X (Kernel Principal Components, TFIDF)")
LRCV(Xkpc_tfidf, y, 5)
print("X (Kernel Principal Components, TFIDF)")
LRCV(Xkpc_tfidf, y, 5, penalty = 'l1')


print("X (Original)")
EnetCV(Xorig, y, 5)
print("X (Original)")
EnetCV(Xorig, y, 5, l1_ratio = 0.85)
print("X (Original, 2-gram)")
EnetCV(X2gram, y, 5)
print("X (Original, TFIDF)")
EnetCV(Xtfidf, y, 5)
print("X (Kernel Principal Components)")
EnetCV(Xkpc, y, 5)
print("X (Kernel Principal Components)")
EnetCV(Xkpc, y, 5, l1_ratio = 0.85)
print("X (Kernel Principal Components, 2-gram)")
EnetCV(Xkpc_2g, y, 5)
print("X (Kernel Principal Components, 2-gram)")
EnetCV(Xkpc_2g, y, 5, l1_ratio = 0.85)
print("X (Kernel Principal Components, TFIDF)")
EnetCV(Xkpc_tfidf, y, 5)
print("X (Kernel Principal Components, TFIDF)")
EnetCV(Xkpc_tfidf, y, 5, l1_ratio = 0.85)



