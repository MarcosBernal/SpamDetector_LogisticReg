#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 12:36:18 2017

@author: rusiano -
"""

import math, time
import numpy as np
import matplotlib.pyplot as plt

# safe log: log(0) = 0
def safe_log(x):
    if (x == 0):
        return 0
    
    return math.log(x)

# safe division: x / 0 = 0    
def safe_division(x,y):
    if (y == 0):
        return 0
    
    return x/y

# given a tuple of sub-rdds and the cross-validation fold index,
#  this method returns a tuple containing training and validation rdds for the
#  corresponding cross-validation fold (the idea is to use one sub-rdd for
#  validation and the remaining ones for training)
def get_train_validation_rdds(sub_rdds, k):
    
    indices=list(range(0, 4))
    
    # choose th validation set to be the k-th sub-rdd
    validation_rdd = sub_rdds[indices.pop(k)]
    
    # initialize the train rdd with the first sub-rdd left
    # and then append all the remaining sub-rdds to it
    train_rdd = sub_rdds[indices.pop(0)]
    for i in indices:
        train_rdd = train_rdd.union(sub_rdds[i])
    
    # save train and validation set in a file
    #validation_rdd.saveAsTextFile('spam.validation' + str(k+1) + '.norm.data')
    #train_rdd.saveAsTextFile('spam.train' + str(k+1) + '.data')
    
    return train_rdd.cache(), validation_rdd.cache()

# Returns the simple sigmoid function output
# x is the features vector without label (x0 is always 1)
# w is the weights vector (w0 is the bias)
def predict(W, X):
    try:
        return 1 / (1 + math.exp(-(np.dot(W, X))))
    except OverflowError:
        return 0

# Update of the cost function for one single data point
def get_cost_upd(y_yhat):
    y, yhat = y_yhat
    return y * safe_log(yhat) + (1-y) * safe_log(1-yhat)

# Update of the weights for one single feature
# when j = 0, X[j] = 1, so we get (yhat - y), i.e the update for the bias
def get_weight_upd(X_y_yhat, j):
    X, y, yhat = X_y_yhat
    return (yhat - y) * X[j]


# Given the ((d,f), (X, y)) tuple, it appends the hypothesis value h to the value tuple
def append_hypothesis(d_f__X_y, W):
    (d, f) = d_f__X_y[0]    # the pair (d,f) is the key
    (X, y) = d_f__X_y[1]    # the pair (feats vector, true label) is the value
    
    return ((d, f), (X, y) + (predict(W[d][f], X), ))

# After having already computed the sum of the cost updates, this method computes
#  the final value of the cost function
def get_train_cost(d_f__costsum, W, M, trainRdd_sizes):
    
    (d, f) = d_f__costsum[0]    # the pair (d,f) is the key
    costsum = d_f__costsum[1]   # the sum of the cost updates is the value
    
    lambdareg = M[d][1]     # lambda depends on the model d
    m = trainRdd_sizes[f]   # train/validation size depends on the fold f
    
    return ((d, f), - costsum/m + lambdareg/(2*m) * np.sum(np.square(W[d][f][1: ])))

# For a given data point, this method returns n new key, value tuples with n being
#  the number of features and the value being the weight update for the j-th weight
def expand_features(d_f__X_y_h, n_feats):
    (d, f) = d_f__X_y_h[0]
    X_y_h = d_f__X_y_h[1]
    return [((d, f, j), get_weight_upd(X_y_h, j)) for j in range(n_feats)]


## Uses the 
#def get_cost(data_rdd, W, lambdareg=0):
#    m = data_rdd.count()
#    
#    #FIRST STEP: compute the predictions for the given weights and append them to the rest
#    # REMEMBER: every row of the rdd is now a tuple (feature_vector, true_label)
#    predictions_rdd = data_rdd\
#    .map(lambda X_y: X_y + (predict(W, X_y[0]),))\
#    .cache()
#    
#    cost = predictions_rdd\
#        .map(lambda X_y_yhat: get_cost_upd(X_y_yhat[1:]))\
#        .reduce(lambda upd1, upd2: upd1 + upd2)
#    
#    cost = - cost/m + ( lambdareg/(2*m) * np.dot(W, W) )
#    
#    return cost
#def gradient_descent(train_rdd, n_epochs=1000, alpha0=0.1, lambdareg=0, decay=0, 
#                     plot=True, seed=123):
#    
#    m = train_rdd.count()
#    n = len(train_rdd.first()[0]) 
#    np.random.seed(seed); new_W = np.random.rand(n)
#    
#    cost_history = []
#    
#    start = time.time()
#    
#    for epoch in range(n_epochs):
#    
#        W = new_W
#        alpha = alpha0 / (1.0 + decay * epoch)
#        reg = (1 - alpha / m * lambdareg)
#    
#        #FIRST STEP: compute the predictions for the given weights and append them to the rest
#        # REMEMBER: every row of the rdd is now a tuple (feature_vector, true_label)
#        predictions_rdd = train_rdd\
#        .map(lambda X_y: X_y + (predict(W, X_y[0]),))
#    
#        #SECOND STEP: compute the total cost for the computed predictions
#        cost = predictions_rdd\
#        .map(lambda X_y_yhat: get_cost_upd(X_y_yhat[1:]))\
#        .reduce(lambda upd1, upd2: upd1 + upd2)
#    
#        cost = - cost/m + ( lambdareg/(2*m) * np.dot(W, W) )
#        
#        cost_history.append(cost)
#        
#        if (epoch % 50 == 0):
#            print("(", epoch, ") Cost: ", cost)
#    
#        #THIRD STEP: update all the weights simoultaneously
#        W_upds = predictions_rdd\
#        .flatMap(lambda X_y_yhat: [(j, get_weight_upd(X_y_yhat, j))
#                                   for j in range(n)])\
#        .reduceByKey(lambda upd1, upd2: upd1 + upd2)\
#        .sortByKey(True)\
#        .values()\
#        .collect()
#        
#        new_W = [(w * reg - alpha / m * w_upd) for w, w_upd in zip(W, W_upds)]
#        
#    end = time.time()
#    
#    if (plot):
#        plt.plot(list(range(len(cost_history))), cost_history)
#    
#    print("> Epochs: ", epoch+1)
#    print("> Gradient Descent Running time: ", ((end-start)/60), "mins")
#    
#    return (cost, W)