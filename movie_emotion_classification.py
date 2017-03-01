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


### Data Set 2: Sentiment Analysis (2-class Classification)
# Read files
file = ".../emotion_data_X.txt"
with open(file) as f:
    reviews2 = f.readlines()

# Create Term Document Matrices
#TDM w/ each word as columns (removing any words freq < min_df
#and freq > 80% of reviews
vc = sklearn.feature_extraction.text.CountVectorizer(min_df=3, max_df=0.9)
Xorig2 = vc.fit_transform(reviews2).toarray()

#TDM w/ 2-gram
vc_2gram = sklearn.feature_extraction.text.CountVectorizer(min_df=3, max_df=0.9, ngram_range=(1, 3))
X2gram2 = vc_2gram.fit_transform(reviews2).toarray()

#TDM w/ TF-IDF (instead of counts)
vc_tfidf = sklearn.feature_extraction.text.TfidfVectorizer(min_df=3, max_df=0.9)
Xtfidf2 = vc_tfidf.fit_transform(reviews2).toarray()

# Emotion
file = ".../y.txt"
with open(file) as f:
    y2 = f.readlines()
y2 = np.asarray(y2)

# Dimension Reduction

##Principal Components
pca = PCA(n_components=20) 
#pca.fit(Xorig2)

Xpc2 = pca.fit_transform(Xorig2)
Xpc_2g2 = pca.fit_transform(X2gram2)
Xpc_tfidf2 = pca.fit_transform(Xtfidf2)


##Kernel Principal Components
kpca = KernelPCA(kernel="rbf", fit_inverse_transform=True, gamma=10)

Xkpc2 = kpca.fit_transform(Xorig2)
Xkpc_2g2 = kpca.fit_transform(X2gram2)
Xkpc_tfidf2 = kpca.fit_transform(Xtfidf2)


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

        #train_cm += confusion_matrix(y_train, yhat_train)
        #test_cm += confusion_matrix(y_test, yhat_test)

    with open("nb_error2.dat", "w") as f:
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
    #print('Training Confusion Matrix')
    #print(train_cm)
    #print('')
    #print('Testing Confusion Matrix')
    #print(test_cm)
    #print('')
    #print('')
    

# Logistic Regression with cross validation 
def LRCV(X, y, K, penalty='l2'):

    lr = LogisticRegression(penalty = penalty)
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

        #train_cm += confusion_matrix(y_train, yhat_train)
        #test_cm += confusion_matrix(y_test, yhat_test)
        
    with open("lr_error2.dat", "w") as f:
        for k in range(K):
            print(trainerror[k], end=", ", file=f)
            print(testerror[k], end=", ", file=f)
        print(file=f)

    print('')
    print('Logistic Regression: penalty = ', penalty)
    print('Cross Validation (', K, '-Fold) Complete', sep='')
    print('')
    print('Training Error:', ['%.4f' % i for i in trainerror])
    print('Training SD:', ['%.4f' % np.std(trainerror)])
    print('Testing Error:', ['%.4f' % i for i in testerror])
    print('Testing SD:', ['%.4f' %np.std(testerror)])
    print('Average Training Error:', round(avg_trainerror, 4))
    print('Average Testing Error:', round(avg_testerror, 4))
    print('')
    #print('Training Confusion Matrix')
    #print(train_cm)
    #print('')
    #print('Testing Confusion Matrix')
    #print(test_cm)
    #print('')
    #print('')



    

#######################################################################
print("X (Original)")
NBCV(Xorig2, y2, 5)
print("X (Original, 3-gram)")
NBCV(X2gram2, y2, 5)
print("X (Original, TFIDF)")
NBCV(Xtfidf2, y2, 5)
print("X (Principal Components)")
NBCV(Xpc2, y2, 5)
print("X (Principal Components, 3-gram)")
NBCV(Xpc_2g2, y2, 5)
print("X (Principal Components, TFIDF)")
NBCV(Xpc_tfidf2, y2, 5)
print("X (Kernel Principal Components)")
NBCV(Xkpc2, y2, 5)
print("X (Kernel Principal Components, 3-gram)")
NBCV(Xkpc_2g2, y2, 5)
print("X (Kernel Principal Components, TFIDF)")
NBCV(Xkpc_tfidf2, y2, 5)




