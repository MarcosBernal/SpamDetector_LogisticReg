
# coding: utf-8
import numpy as np
import time
from pyspark import SparkContext, SparkConf
from marcoslib import normalize_data
import maxlib as mxl

    
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
HyperParams = sorted([(np.random.uniform(low=min_alpha0, high=max_alpha0),
          np.random.uniform(low=min_lambdareg, high=max_lambdareg)) for _ in range(n_models)])

print(">>> MODELS: ")
print(dict(zip(range(n_models), HyperParams)))

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
    .map(lambda d_f__Xy: mxl.append_hypothesis(d_f__Xy, W))\
    .cache() # <<<<< important to cache!
    

    #SECOND 2: compute the total cost for the new predictions 
    # for every model d and every fold f
    trainCosts_dict = hypothesis_rdd\
    .mapValues(lambda X_y_h: mxl.get_cost_upd(X_y_h[1:]))\
    .reduceByKey(lambda df_upd1, df_upd2: df_upd1 + df_upd2).sortByKey(True)\
    .map(lambda d_f__costsum: mxl.get_train_cost(d_f__costsum, W, HyperParams, trainRdd_sizes))\
    .collectAsMap()
    
    # append the last training costs to the history list
    for (d, f) in trainCosts_dict:
        trainCosts_history[d][f].extend([trainCosts_dict[(d, f)]])
    
    
    # STEP 3: 
    # - for every model and fold vector, create 1 new for every feature
    # - compute all the weights updates simoultaneously for all the models, folds and features
    # - get the updates in the form of a dictionary
    Wupds = hypothesis_rdd\
    .flatMap(lambda d_f__X_y_h: mxl.expand_features(d_f__X_y_h, n))\
    .reduceByKey(lambda dfj_upd1, dfj_upd2: dfj_upd1 + dfj_upd2).sortByKey(True)\
    .collectAsMap()
    
    # update the weights for every model, fold, and feature in the dictionary
    for (d, f, j) in Wupds:
        alpha = HyperParams[d][0]
        lambdareg = HyperParams[d][1]
        #m = trainRdd_sizes[f]
        W[d][f][j] = W[d][f][j] * (1 - alpha * lambdareg) - alpha * Wupds[(d, f, j)]
        
    epoch_end = time.time()
    
    print(epoch, " (", epoch_end - epoch_start, "s)")

        
end = time.time()
print("Total Gradient-Descent Running Time: ", (end-start)/60, "mins")
        
#%% VALIDATION

kfoldModelsCosts_dict = validRddByModelAndFold\
.map(lambda d_f__Xy: mxl.append_hypothesis(d_f__Xy, W))\
.mapValues(lambda X_y_h: mxl.get_cost_upd(X_y_h[1:]))\
.reduceByKey(lambda df_upd1, df_upd2: df_upd1 + df_upd2).sortByKey(True)\
.map(lambda d_f__costsum: mxl.get_train_cost(d_f__costsum, W, HyperParams, validRdd_sizes))\
.map(lambda df_cost: (df_cost[0][0], df_cost[1]))\
.reduceByKey(lambda d_cost1, d_cost2: d_cost1+d_cost2)\
.mapValues(lambda validation_cost_sum: validation_cost_sum / k)\
.collectAsMap()

end = time.time()
best_model = min(kfoldModelsCosts_dict, key=kfoldModelsCosts_dict.get)

print("Total Cross-Validation Elapsed Time: ", (end-start)/60, "mins")
print("Cross Validation Errors: ", kfoldModelsCosts_dict)
print("Best Model: ", best_model, 
      "(alpha: ", HyperParams[best_model][0],", lambda: ", HyperParams[best_model][1], ")")

#%% TRAINING SELECTED MODEL

m = non_test_rdd.count()
alpha = HyperParams[best_model][0]
lambdareg = HyperParams[best_model][1]
bestModel_W = [np.random.uniform(-3,3) for _ in range(n)]

for epoch in range(n_epochs):

    epoch_start = time.time()
    
    # STEP 1: compute predictions with the new weights 
    # and append them to the original rdd
    bestModel_hypothesis = non_test_rdd\
    .map(lambda X_y: X_y + (mxl.predict(bestModel_W, X_y[0]), ))\
    .cache() # <<<<< important to cache!
    

    #SECOND 2: compute the total cost for the new predictions 
    # for every model d and every fold f
    bestModel_trainCost = - bestModel_hypothesis\
    .map(lambda X_y_h: mxl.get_cost_upd(X_y_h[1:]))\
    .reduce(lambda upd1, upd2: upd1 + upd2) / m + ( lambdareg / (2*m) * np.sum(np.square(bestModel_W[1:])))
    
    
    # STEP 3: 
    # - for every model and fold vector, create 1 new for every feature
    # - compute all the weights updates simoultaneously for all the models, folds and features
    # - get the updates in the form of a dictionary
    bestModel_Wupds = bestModel_hypothesis\
    .flatMap(lambda X_y_yhat: [(j, mxl.get_weight_upd(X_y_yhat, j)) for j in range(n)])\
    .reduceByKey(lambda upd1, upd2: upd1 + upd2).sortByKey(True)\
    .values().collect()
    
    bestModel_W = [(w * (1 - alpha * lambdareg) - alpha / m * wupd) for w, wupd in zip(bestModel_W, bestModel_Wupds)]
        
    epoch_end = time.time()
    
    print(epoch, " (", epoch_end - epoch_start, "s)")

        
end = time.time()

#%% TESTING

m = test_rdd.count()

yh = test_rdd.map(lambda X_y: (X_y[1], mxl.predict(bestModel_W, X_y[0]))).cache()

tp = yh.filter(lambda y_h: y_h[0] == 1 and y_h[1] == 1)
tn = yh.filter(lambda y_h: y_h[0] == 0 and y_h[1] == 0)
fp = yh.filter(lambda y_h: y_h[0] == 0 and y_h[1] == 1)
fn = yh.filter(lambda y_h: y_h[0] == 1 and y_h[1] == 0)

accuracy = (tp + tn) / test_rdd.count()
precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1 = 2 * (precision * recall) / (precision + recall)
