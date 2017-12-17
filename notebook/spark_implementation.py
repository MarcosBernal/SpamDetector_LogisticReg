#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math
from rdd_creation_and_stats import sc

# coeff and values must be lists with the same length
def predict(coeff, values):

    logit = sum([ coeff[index] * values[index] for index in range(len(coeff)) ]) + coeff[-1] # Last element is bias(coeff) - label(values)
    
    if logit < 0: # when logit becomes a large positive value, math.exp(gamma) overflows
        sigmoid = 1 - 1 / (1 + math.exp(logit))
    else:
        sigmoid = 1 / (1 + math.exp(-logit))
    return sigmoid


def training_logistic_regression_model(training_set, l_rate, iterations=1000, lambda_reg = 0, show_iteration=0):
    
    if show_iteration > iterations:
        show_iteration = iterations-1
    
    # Init the coeff
    n_features = len(training_set[0])
    
    weights = [0.0 for index in range(n_features)]
    l_rate_over_m = l_rate/len(training_set)
    
    # See http://holehouse.org/mlclass/07_Regularization.html for details
    for iteration in range(iterations):
        sum_error = 0
        weights_upd = [ [] for index in range(n_features)]
        
        for instance in training_set:
            y = instance[-1]   # the last element of each row is the true label
            yhat = predict(coeff=weights, values=instance)  # compute the predicted label according to current weights
            error = yhat - y
            
            sum_error += error**2
            
            for i in range(len(instance)): # Adding items of sumatorium 
                if (i < len(instance) -1):
                    weights_upd[i].append((error * instance[i]))
                else:
                    weights_upd[-1].append(error) # Calculating the bias \theta_0 
        
        # Regularization
        for i in range(n_features):  
            weights[i] = weights[i] * (1 - l_rate_over_m * lambda_reg) - (l_rate_over_m * sum(weights_upd[i]))
            
        if(show_iteration > 0 and iteration%show_iteration == 0):
            cost =  cost_function_spark(weights, sc.parallelize(training_set), lambda_reg)
            print('>iteration=',iteration,' lrate=',l_rate, ' error=', sum_error, 'cost=', cost)
        #print(predict(coeff,training_set[0]), training_set[0][-1], " - ",predict(coeff,training_set[-1]), training_set[-1][-1])

        
    return ( sum_error, weights )


def get_weight_upd(i_feature, weights, instance):
    n_features = len(instance)
    if (i_feature < n_features -1):
        return (predict(weights, instance) - instance[-1]) * instance[i_feature]
    else:
        return (predict(weights, instance) - instance[-1])
    
    
def update_weight(weight_weightupd, l_rate_over_size, lambda_reg):
    weight, weight_upd = weight_weightupd
    return (weight * (1 - l_rate_over_size * lambda_reg)) - (l_rate_over_size * weight_upd)

    
def get_cost_upd(weights, instance):
    y = instance[-1]                    # true label
    yhat = predict(weights, instance)   # predicted label
    
    return y * math.log(yhat) + (1 - y) * math.log(1 - yhat)


## SPARK IMPLEMENTATIONS ###############################################################################################

# https://www.coursera.org/learn/machine-learning/lecture/10sqI/map-reduce-and-data-parallelism

def cost_function_spark(weights, data_rdd, lambda_reg=0):

    size = data_rdd.count()
        
    summation = data_rdd \
    .map(lambda instance: get_cost_upd(weights, instance)) \
    .reduce(lambda cost_upd1, cost_upd2: cost_upd1 + cost_upd2)

    reg_term = lambda_reg/size * sum([ weights[index]**2 for index in range(len(weights)-1) ]) 
    cost = -summation/size + reg_term
    return cost 


def training_logistic_regression_model_spark(train_rdd, l_rate, iterations=1000, lambda_reg = 0, show_iteration=0):

    # compute useful constants for further computations
    l_rate_over_size = l_rate / train_rdd.count()
    n_features = len(train_rdd.first())

    # initialize the weights vector (one weight per feature)
    weights = [0.0 for index in range(n_features)]
     
    for iteration in range(iterations):
        
        # the key of the <key, value> pairs is the index of the feature, 
        #  so that we can reduce by it and sum all the updates
        weights = train_rdd \
        .flatMap(lambda instance: [(f, (weights[f], get_weight_upd(f, weights, instance))) for f in range(n_features)])\
        .reduceByKey(lambda x, y: (x[0], x[1] + y[1]))\
        .map(lambda key_weight_weightupd: update_weight(weight_weightupd=key_weight_weightupd[1],
                                                        l_rate_over_size=l_rate_over_size, 
                                                        lambda_reg=0))\
        .collect()
        
        cost =  cost_function_spark(weights, train_rdd, lambda_reg)
        
        if(show_iteration > 0 and iteration%show_iteration == 0):
            cost =  cost_function_spark(weights, train_rdd, lambda_reg)
            print('>iteration=',iteration,' lrate=',l_rate, ' cost_func=', cost)
            
    return cost, weights


def optimal_learning_rate_value(training_set, initial_l_rate = 0.05, list_size = 10, max_iterations = 20000, lambda_reg = 0, show_iteration=0):
    order_log_list = [initial_l_rate]
    [ order_log_list.append(order_log_list[-1]*(2)) for index in range(list_size)]
    weight_list = []
    for l_rate in order_log_list:
        weight_list.append(training_logistic_regression_model(training_set, l_rate, max_iterations, lambda_reg, show_iteration))
    return sorted(weight_list) 


def optimal_learning_rate_value_spark(training_set_rdd, initial_l_rate = 0.05, list_size = 10, max_iterations = 20000, lambda_reg = 0, show_iteration=0):
    order_log_list = [initial_l_rate]
    [ order_log_list.append(order_log_list[-1]*(2)) for index in range(list_size)]
    weight_list = []
    for l_rate in order_log_list:
        weight_list.append(training_logistic_regression_model_spark(training_set_rdd, l_rate, max_iterations, lambda_reg, show_iteration))
    return sorted(weight_list)