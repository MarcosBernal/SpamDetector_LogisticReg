
# coding: utf-8
import numpy as np
import time
from pyspark import SparkContext, SparkConf
from marcoslib import normalize_data
from maxlib import get_train_validation_rdds, gradient_descent, get_cost, predict, get_cost_upd, get_weight_upd

    
n_executors_spark = 2
conf = SparkConf()\
.setAppName("Spam Filter")\
.setMaster("local["+str(n_executors_spark)+"]")\
.set("spark.hadoop.validateOutputSpecs", "false");

sc =   SparkContext.getOrCreate(conf=conf)

file_object  = open('spam.data', 'r')
lines = file_object.readlines()
file_object.close()
    
total_size = len(lines)


# Creating RDD
master_rdd = sc.parallelize(lines)\
.map(lambda x: [float(item) for item in x.split('\n')[0].split(" ")])
master_norm_rdd = normalize_data(master_rdd)

# split each data vector into train vector and label
# append a 1 at the head of the array for further computation
master_norm_rdd = master_norm_rdd.map(lambda dv: ([1] + dv[:-1], dv[-1]))

# divide the original rdd in non-test and test rdds
non_test_rdd, test_rdd = master_norm_rdd.randomSplit([0.8, 0.2])

# save test set in a file
#test_rdd.saveAsTextFile('spam.test.set')

########## PARALLELIZED CROSS-VALIDATION #####################################
#%% CONSTANTS
k = 4  # k-fold iterations

#%% MODEL DEFINITION (HYPERPARAMETERS)

alpha0 = 0.5
lambdareg = 0
decay = 0.005

#%% CROSS VALIDATION SPLIT

kfold_rdd = non_test_rdd\
.flatMap(lambda dv: [(k, dv) for k in range(k)]).sortByKey(True)\
.zipWithIndex().map(lambda kdv_i: (kdv_i[0][0], (kdv_i[1] % k, ) + kdv_i[0][1]))

kfold_train_rdd = kfold_rdd\
.filter(lambda k_idv: k_idv[0] != k_idv[1][0])\
.mapValues(lambda i_dv: i_dv[1:])\
.cache()

kfold_validation_rdd = kfold_rdd\
.filter(lambda k_idv: k_idv[0] == k_idv[1][0])\
.mapValues(lambda i_dv: i_dv[1:])\
.cache()

n = len(non_test_rdd.first()[0])
m = kfold_train_rdd.count() / k

kWs = [np.random.rand(n) for _ in range(k)]
kfold_train_costs_history = [[] for _ in range(k)]

#%% TRAINING (GRADIENT DESCENT)

start = time.time()
for epoch in range(400):

    epoch_start = time.time()
    
    alpha = alpha0 / (1.0 + decay * epoch)
    reg = (1 - alpha / m * lambdareg)
    
    # STEP 1: compute predictions with the new weights
    kfold_predictions_rdd = kfold_train_rdd\
    .map(lambda k_Xy: (k_Xy[0], k_Xy[1] + (predict(kWs[k_Xy[0]], k_Xy[1][0]), )))\
    .cache() # <<<<< super important to cache!
    

    #SECOND 2: compute the total cost for the new predictions
    kfold_train_costs = kfold_predictions_rdd\
    .mapValues(lambda X_y_yhat: get_cost_upd(X_y_yhat[1:]))\
    .reduceByKey(lambda k_upd1, k_upd2: k_upd1 + k_upd2).sortByKey(True)\
    .map(lambda k_costsum: (k_costsum[0], 
        - k_costsum[1]/m + lambdareg/(2*m) * np.sum(np.square(kWs[k_costsum[0]]))))\
    .collectAsMap()
    
    for k in kfold_train_costs:
        kfold_train_costs_history[k].extend([kfold_train_costs[k]])
    
    
    # STEP 3: update all the weights simoultaneously 
    kWupds = kfold_predictions_rdd\
    .flatMap(lambda k_Xyyhat: [((k_Xyyhat[0], j), get_weight_upd(k_Xyyhat[1], j)) for j in range(n)])\
    .reduceByKey(lambda kj_upd1, kj_upd2: kj_upd1 + kj_upd2).sortByKey(True)
    
    kWupds_dict = kWupds.collectAsMap()  
    for (k, j) in kWupds_dict:
        kWs[k][j] = kWs[k][j] * reg - alpha / m * kWupds_dict[(k, j)]
        
    epoch_end = time.time()
    
    if (epoch % 50 == 0):
        print("(",epoch,") Gradient Descent running time: ", (epoch_end - epoch_start)/60, "mins")
        
        
#%% VALIDATION

kfold_cost = kfold_validation_rdd\
.map(lambda k_Xy: (k_Xy[0], k_Xy[1] + (predict(kWs[k_Xy[0]], k_Xy[1][0]), )))\
.mapValues(lambda X_y_yhat: get_cost_upd(X_y_yhat[1:]))\
.reduceByKey(lambda k_upd1, k_upd2: k_upd1 + k_upd2).sortByKey(True)\
.map(lambda k_costsum: (k_costsum[0], 
        - k_costsum[1]/m + lambdareg/(2*m) * np.sum(np.square(kWs[k_costsum[0]]))))\
.values().reduce(lambda a, b: a+b) / k

end = time.time()
print("Total Cross-Validation Elapsed Time: ", (end-start)/60, "mins")
print("Cross Validation Error: ", kfold_cost)


# divide the non-test rdd in 4 sub-rdds
#sub_rdds = non_test_rdd.randomSplit(np.repeat(1.0/n_folds, n_folds))

#folds_rdd = sc.parallelize(list(range(n_folds)))
#
#start = time.time()
#kfold_cost = folds_rdd\
#.map(lambda k: (k, get_train_validation_rdds(sub_rdds, k)))\
#.map(lambda k_trainval: (k, gradient_descent(get_train_validation_rdds(sub_rdds, k)[0], 
#                                    n_epochs=400, alpha=0.1)[0]))\
#.reduce(lambda kfold_cost1, kfold_cost2: kfold_cost1 + kfold_cost2)
#
#kfold_cost \= n_folds
#end = time.time()


#

#start = time.time()
#kfold_cost = 0
#for k in range(0, n_folds):
#    iteration_start = time.time()
#    print("=== Fold ", k+1, " ====================================")
#    
#    # for every iteration get a different train and validation sets
#    train_rdd, validation_rdd = get_train_validation_rdds(sub_rdds, k)
#    
#    # train the model and get some weights
#    train_cost, train_W = gradient_descent(train_rdd, n_epochs=400, alpha0=0.3)
#    
#    # validate the model and get the cost (to be averaged)
#    validation_cost = get_cost(validation_rdd, train_W) 
#    kfold_cost += validation_cost
#    
#    iteration_end = time.time()
#
#    print("> Train/Validation Cost: ", train_cost, "/", validation_cost)
#    print("> Total Elapsed time: ", (iteration_end-iteration_start)/60, "mins")
#    print()
#    
#kfold_cost /= n_folds
#end = time.time()
#
#print("=== ", n_folds, "-fold Cross Validation =================")
#print("> Avg cost: ", kfold_cost)
#print("> Elapsed time: ", ((end-start)/60), "mins")




