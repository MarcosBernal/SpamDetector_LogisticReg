
# coding: utf-8
import numpy as np
import time
from pyspark import SparkContext, SparkConf
from marcoslib import normalize_data
from maxlib import append_hypothesis, expand_features, get_train_cost, get_cost_upd

    
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
master_norm_rdd = master_norm_rdd.map(lambda v: ([1] + v[:-1], v[-1]))

# divide the original rdd in non-test and test rdds
non_test_rdd, test_rdd = master_norm_rdd.randomSplit([0.8, 0.2])

# save test set in a file
#test_rdd.saveAsTextFile('spam.test.set')

########## PARALLELIZED CROSS-VALIDATION #####################################
#%% CONSTANTS
n_models = 10
k = 4  # k-fold iterations
n_epochs = 250
n = len(non_test_rdd.first()[0]) # number of features
    
min_alpha0 = 0.001; max_alpha0 = 0.01
min_lambdareg = 0.0; max_lambdareg = 0.5

#%% MODEL DEFINITION (HYPERPARAMETERS)
M = sorted([(np.random.uniform(low=min_alpha0, high=max_alpha0),
          np.random.uniform(low=min_lambdareg, high=max_lambdareg)) for _ in range(n_models)])

print(">>> MODELS: ")
print(dict(zip(range(n_models), M)))

rddByFold = non_test_rdd\
.zipWithIndex().map(lambda Xy__i: (Xy__i[1] % k, Xy__i[0])).sortByKey(True)\
.flatMap(lambda i__Xy: [((f, i__Xy[0]), i__Xy[1]) for f in range(k)])\
.cache()

trainRddByFold = rddByFold\
.filter(lambda f_i__Xy: f_i__Xy[0][0] != f_i__Xy[0][1])\
.map(lambda f_i__Xy: (f_i__Xy[0][0], f_i__Xy[1]))

validRddByFold = rddByFold\
.filter(lambda f_i__Xy: f_i__Xy[0][0] == f_i__Xy[0][1])\
.map(lambda f_i__Xy: (f_i__Xy[0][0], f_i__Xy[1]))

trainRdd_sizes = [len(res) for fold, res in (trainRddByFold.groupByKey().sortByKey(True).collect())]
validRdd_sizes = [len(res) for fold, res in (validRddByFold.groupByKey().sortByKey(True).collect())]

trainRddByModelAndFold = trainRddByFold\
.flatMap(lambda f__Xy: [((d, f__Xy[0]), f__Xy[1]) for d in range(n_models)])\
.cache()

validRddByModelAndFold = validRddByFold\
.flatMap(lambda f__Xy: [((d, f__Xy[0]), f__Xy[1]) for d in range(n_models)])\
.cache()

#%% CROSS VALIDATION SPLIT

# instantiate a Weights list containing one array for each model
# each array corresponds to the most updated weights vector for a given model
np.random.seed(123)
W = [[[np.random.uniform(-3, 3) for _ in range(n)] for _ in range(k)] for _ in range(n_models)]

# instantiate a cost hisotry list containing one array for each model
# each array corresponds to the most updated history of training costs for a given model
trainCosts_history = [[[] for _ in range(k)] for _ in range(n_models)]

  
#%% TRAINING (GRADIENT DESCENT)

start = time.time()
for epoch in range(n_epochs):

    epoch_start = time.time()
    
    # STEP 1: compute predictions with the new weights 
    # and append them to the original rdd
    hypothesis_rdd = trainRddByModelAndFold\
    .map(lambda d_f__Xy: append_hypothesis(d_f__Xy, W))\
    .cache() # <<<<< important to cache!
    

    #SECOND 2: compute the total cost for the new predictions 
    # for every model d and every fold f
    trainCosts_dict = hypothesis_rdd\
    .mapValues(lambda X_y_h: get_cost_upd(X_y_h[1:]))\
    .reduceByKey(lambda df_upd1, df_upd2: df_upd1 + df_upd2).sortByKey(True)\
    .map(lambda d_f__costsum: get_train_cost(d_f__costsum, W, M, trainRdd_sizes))\
    .collectAsMap()
    
    # append the last training costs to the history list
    for (d, f) in trainCosts_dict:
        trainCosts_history[d][f].extend([trainCosts_dict[(d, f)]])
    
    
    # STEP 3: 
    # - for every model and fold vector, create 1 new for every feature
    # - compute all the weights updates simoultaneously for all the models, folds and features
    # - get the updates in the form of a dictionary
    Wupds = hypothesis_rdd\
    .flatMap(lambda d_f__X_y_h: expand_features(d_f__X_y_h, n))\
    .reduceByKey(lambda dfj_upd1, dfj_upd2: dfj_upd1 + dfj_upd2).sortByKey(True)\
    .collectAsMap()
    
    # update the weights for every model, fold, and feature in the dictionary
    for (d, f, j) in Wupds:
        alpha = M[d][0]
        lambdareg = M[d][1]
        #m = trainRdd_sizes[f]
        W[d][f][j] = W[d][f][j] * (1 - alpha * lambdareg) - alpha * Wupds[(d, f, j)]
        
    epoch_end = time.time()
    
    print(epoch, " (", epoch_end - epoch_start, "s)")

        
end = time.time()
print("Total Gradient-Descent Running Time: ", (end-start)/60, "mins")
        
#%% VALIDATION

kfoldModelsCosts_dict = validRddByModelAndFold\
.map(lambda d_f__Xy: append_hypothesis(d_f__Xy, W))\
.mapValues(lambda X_y_h: get_cost_upd(X_y_h[1:]))\
.reduceByKey(lambda df_upd1, df_upd2: df_upd1 + df_upd2).sortByKey(True)\
.map(lambda d_f__costsum: get_train_cost(d_f__costsum, W, M, validRdd_sizes))\
.map(lambda df_cost: (df_cost[0][0], df_cost[1]))\
.reduceByKey(lambda d_cost1, d_cost2: d_cost1+d_cost2)\
.mapValues(lambda validation_cost_sum: validation_cost_sum / k)\
.collectAsMap()

end = time.time()
best_model = min(kfoldModelsCosts_dict, key=kfoldModelsCosts_dict.get)

print("Total Cross-Validation Elapsed Time: ", (end-start)/60, "mins")
print("Cross Validation Errors: ", kfoldModelsCosts_dict)
print("Best Model: ", best_model, 
      "(alpha: ", M[best_model][0],", lambda: ", M[best_model][1], ")")

#%% TRAINING SELECTED MODEL

for epoch in range(n_epochs):

    epoch_start = time.time()
    
    # STEP 1: compute predictions with the new weights 
    # and append them to the original rdd
    hypothesis_rdd = trainRddByModelAndFold\
    .map(lambda d_f__Xy: append_hypothesis(d_f__Xy, W))\
    .cache() # <<<<< important to cache!
    

    #SECOND 2: compute the total cost for the new predictions 
    # for every model d and every fold f
    trainCosts_dict = hypothesis_rdd\
    .mapValues(lambda X_y_h: get_cost_upd(X_y_h[1:]))\
    .reduceByKey(lambda df_upd1, df_upd2: df_upd1 + df_upd2).sortByKey(True)\
    .map(lambda d_f__costsum: get_train_cost(d_f__costsum, W, M, trainRdd_sizes))\
    .collectAsMap()
    
    # append the last training costs to the history list
    for (d, f) in trainCosts_dict:
        trainCosts_history[d][f].extend([trainCosts_dict[(d, f)]])
    
    
    # STEP 3: 
    # - for every model and fold vector, create 1 new for every feature
    # - compute all the weights updates simoultaneously for all the models, folds and features
    # - get the updates in the form of a dictionary
    Wupds = hypothesis_rdd\
    .flatMap(lambda d_f__X_y_h: expand_features(d_f__X_y_h, n))\
    .reduceByKey(lambda dfj_upd1, dfj_upd2: dfj_upd1 + dfj_upd2).sortByKey(True)\
    .collectAsMap()
    
    # update the weights for every model, fold, and feature in the dictionary
    for (d, f, j) in Wupds:
        alpha = M[d][0]
        lambdareg = M[d][1]
        #m = trainRdd_sizes[f]
        W[d][f][j] = W[d][f][j] * (1 - alpha * lambdareg) - alpha * Wupds[(d, f, j)]
        
    epoch_end = time.time()
    
    print(epoch, " (", epoch_end - epoch_start, "s)")

        
end = time.time()


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




