#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 12:36:18 2017

@author: rusiano -
"""

import math, time
import numpy as np
import matplotlib.pyplot as plt

#def reduce_features(data_rdd):
#    
#    is_to_keep = [6:13] + [15:18] + [22:27] + [29, 31, 40, 42, 45, 46, 51] + [53:57] 
#    
#    return data_rdd.map(lambda X_y: ([X_y[0][i] for i in is_to_keep)], X_y[1])

# given a tuple of sub-rdds and the cross-validation iteration index,
#  this method returns a tuple containing training and validation rdds
def get_train_validation_rdds(sub_rdds, k):
    
    indices=list(range(0, 4))
    
    # the validation set is the k-th sub-rdd
    validation_rdd = sub_rdds[indices.pop(k)]
    
    # initialize the train rdd with the first sub-rdd left
    train_rdd = sub_rdds[indices.pop(0)]
    
    # append all the remaining sub-rdds to the train-rdd
    for i in indices:
        train_rdd = train_rdd.union(sub_rdds[i])
    
    # save train and validation set in a file
    validation_rdd.saveAsTextFile('spam.validation' + str(k+1) + '.norm.data')
    train_rdd.saveAsTextFile('spam.train' + str(k+1) + '.data')
    
    return train_rdd.cache(), validation_rdd.cache()


# x is the features vector without label (x0 is always 1)
# w is the weights vector (w0 is the bias)
def predict(W, X, decision_boundary = 0.5):
    return int(1 / (1 + math.exp(-(np.dot(W, X)))) <= decision_boundary)

def get_cost_upd(y_yhat):
    y, yhat = y_yhat
    return y * math.log(yhat) + (1-y) * math.log(1-yhat)

# when j = 0, X[j] = 1, so we get (yhat - y), i.e the update for the bias
def get_weight_upd(X_y_yhat, j):
    X, y, yhat = X_y_yhat
    return (yhat - y) * X[j]


def gradient_descent(train_rdd, n_epochs=1000, alpha0=0.1, lambdareg=0, decay=0, 
                     decision_boundary = 0.5, plot=True, seed=123):
    
    m = train_rdd.count()
    n = len(train_rdd.first()[0]) 
    np.random.seed(seed); new_W = np.random.rand(n)
    
    old_cost = 0
    cost_history = []
    
    start = time.time()
    
    for epoch in range(n_epochs):
    
        W = new_W
        alpha = alpha0 / (1.0 + decay * epoch)
        reg = (1 - alpha / m * lambdareg)
    
        #FIRST STEP: compute the predictions for the given weights and append them to the rest
        # REMEMBER: every row of the rdd is now a tuple (feature_vector, true_label)
        predictions_rdd = train_rdd\
        .map(lambda X_y: X_y + (predict(W, X_y[0], decision_boundary),))\
        .cache()
    
        #SECOND STEP: compute the total cost for the computed predictions
        cost = predictions_rdd\
        .map(lambda X_y_yhat: get_cost_upd(X_y_yhat[1:]))\
        .reduce(lambda upd1, upd2: upd1 + upd2)
    
        cost = - cost/m + ( lambdareg/(2*m) * np.dot(W, W) )
        
        cost_history.append(cost)
        old_cost = cost
        
        if (epoch % 50 == 0):
            print("(", epoch, ") Cost: ", cost)
    
        #THIRD STEP: update all the weights simoultaneously
        W_upds = predictions_rdd\
        .flatMap(lambda X_y_yhat: [(j, get_weight_upd(X_y_yhat, j))
                                   for j in range(n)])\
        .reduceByKey(lambda upd1, upd2: upd1 + upd2)\
        .sortByKey(True)\
        .values()\
        .collect()
        
        new_W = [(w * reg - alpha / m * w_upd) for w, w_upd in zip(W, W_upds)]
        
    end = time.time()
    
    if (plot):
        plt.plot(list(range(len(cost_history))), cost_history)
    
    print("> Epochs: ", epoch+1)
    print("> Gradient Descent Running time: ", ((end-start)/60), "mins")
    
    return (cost, W)


def get_cost(data_rdd, W, lambdareg=0, decision_boundary=0.5):
    m = data_rdd.count()
    
    #FIRST STEP: compute the predictions for the given weights and append them to the rest
    # REMEMBER: every row of the rdd is now a tuple (feature_vector, true_label)
    predictions_rdd = data_rdd\
    .map(lambda X_y: X_y + (predict(W, X_y[0], decision_boundary),))\
    .cache()
    
    cost = predictions_rdd\
        .map(lambda X_y_yhat: get_cost_upd(X_y_yhat[1:]))\
        .reduce(lambda upd1, upd2: upd1 + upd2)
    
    cost = - cost/m + ( lambdareg/(2*m) * np.dot(W, W) )
    
    return cost