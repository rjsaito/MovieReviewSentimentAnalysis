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
path_neg = ".../neg/"
path_pos = ".../pos/"

neg_reviews = [read_txt(join(path_neg , f)) for f in listdir(path_neg)]
pos_reviews = [read_txt(join(path_pos , f)) for f in listdir(path_pos)]
all_reviews = neg_reviews + pos_reviews

# Create Term Document Matrices
#TDM w/ each word as columns (removing any words freq < min_df
#and freq > 80% of reviews
vc = sklearn.feature_extraction.text.CountVectorizer(min_df=5, max_df=0.8)
Xorig = vc.fit_transform(all_reviews).toarray()

print(Xorig.shape)







