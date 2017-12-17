#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time
import spark_implementation as our
from rdd_creation_and_stats import sc, train_validation_rdd#, test_rdd


# Testing sequential and spark version of logistic regression with a linear solvable problem

data = sc.parallelize([
    [0,   0,   0],
    [0.5, 2,   0],
    [1,   1.8, 0],
    [0.5, 1.5, 0],
    [1,   1.5, 0],
    [2,   0.5, 0],
    [0.5, 3,   1],
    [0.5, 3,   1],
    [2,  2,    1],
    [3,  2,    1],
    [3,  1,    1],
    [3,  3,    1]    
])

print("Training model for linear separable problem.")
start = time.time()
estimated_weight = our.training_logistic_regression_model(data.collect(), 0.5, 500, 0.01, show_iteration=100) # Check if code works
end = time.time()
print("Optimal learning rate computed sequentially.")
print(" > Elapsed Time: ", end - start)
print("Optimal: [40, 51, -153] - Computed:", estimated_weight[1])
print("\nPredictions within the training set")

estimated_weight = estimated_weight[1]
#estimated_weight =  # Optimal
for index in range(data.count()):     
    print(our.predict(estimated_weight,data.collect()[index]), data.collect()[index][-1])
    
  
l_rate = 1
list_size = 2
iterations = 100    
lambda_reg = 0.5
show_ite = 50

print("\nTraining SPARK algorithm with ", iterations," iterations", " and ", list_size, "different l_rates")    
start = time.time()
values_spark = our.optimal_learning_rate_value_spark(train_validation_rdd, initial_l_rate=l_rate, list_size=list_size, max_iterations=iterations, lambda_reg=lambda_reg, show_iteration=show_ite)
end = time.time()
print(" > Elapsed Time: ", end - start)

print("Training SEQUENTIAL algorithm with ", iterations," iterations", " and ", list_size, "different l_rates")   
start = time.time()
values = our.optimal_learning_rate_value(train_validation_rdd.collect(), initial_l_rate=l_rate, list_size=list_size, max_iterations=iterations, lambda_reg=lambda_reg, show_iteration=show_ite)
end = time.time()
print(" > Elapsed Time: ", end - start)