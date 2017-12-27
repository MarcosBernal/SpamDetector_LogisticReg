
# coding: utf-8
import numpy as np
import time
from pyspark import SparkContext, SparkConf
from marcoslib import normalize_data
import maxlib as mxl
import matplotlib.pyplot as plt
import math

    
n_executors_spark = 2
conf = SparkConf()\
.setAppName("Spam Filter")\
.setMaster("local["+str(n_executors_spark)+"]")\
.set("spark.hadoop.validateOutputSpecs", "false");

sc =   SparkContext.getOrCreate(conf=conf)

#%%

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
nonTestRdd, testRdd = master_norm_rdd.randomSplit([0.8, 0.2])

# save test set in a file
#testRdd.saveAsTextFile('spam.test.set')

########## PARALLELIZED CROSS-VALIDATION #####################################
#%% CONSTANTS
n_models = 10   # number of models to train
n_folds = 4     # k-fold iterations
n_epochs = 1000  # number of gradient descent iterations
n_feats = len(nonTestRdd.first()[0]) # number of features

#%% MODEL DEFINITION (HYPERPARAMETERS)
min_alpha0 = 0.001; max_alpha0 = 0.004
min_lambdareg = 0.0; max_lambdareg = 3.0

HyperParams = sorted([(np.random.uniform(low=min_alpha0, high=max_alpha0),
                       np.random.uniform(low=min_lambdareg, high=max_lambdareg)) for _ in range(n_models)])

HyperParams = dict(zip(range(n_models), HyperParams))
print(">>> MODELS >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
print("".ljust(13), "alpha".ljust(23), "lambda")
for i in HyperParams:
    print("{0:3}: {1:23} {2:23}".format(i+1, HyperParams[i][0], HyperParams[i][1]))
print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
#%% DATASET SPLIT

nonTestRdd_byFold = nonTestRdd\
.zipWithIndex().map(lambda Xy__i: (Xy__i[1] % n_folds, Xy__i[0])).sortByKey(True)\
.flatMap(lambda i__Xy: [((f, i__Xy[0]), i__Xy[1]) for f in range(n_folds)])\
.cache()

trainRdd_byFold = nonTestRdd_byFold\
.filter(lambda f_i__Xy: f_i__Xy[0][0] != f_i__Xy[0][1])\
.map(lambda f_i__Xy: (f_i__Xy[0][0], f_i__Xy[1]))

validationRdd_ByFold = nonTestRdd_byFold\
.filter(lambda f_i__Xy: f_i__Xy[0][0] == f_i__Xy[0][1])\
.map(lambda f_i__Xy: (f_i__Xy[0][0], f_i__Xy[1]))

trainRdd_foldsSizes = trainRdd_byFold.countByKey()
validRdd_foldsSizes = validationRdd_ByFold.countByKey()

trainRdd_byModelAndFold = trainRdd_byFold\
.flatMap(lambda f__Xy: [((d, f__Xy[0]), f__Xy[1]) for d in range(n_models)])\
.cache()

validationRdd_byModelAndFold = validationRdd_ByFold\
.flatMap(lambda f__Xy: [((d, f__Xy[0]), f__Xy[1]) for d in range(n_models)])\
.cache()

#%% MODEL SELECTION 1 - TRAINING (CROSS VALIDATION w/ GRADIENT DESCENT)

# 1) initialize some random weights for all models and folds considered
# 2) use parallelized gradient descent to iteratively update all the weights
# 3) finally, get "optimal" weights for all models and folds

# instantiate a Weights list containing one array for each model and fold
# each array corresponds to the most updated weights vector for a given model and fold
np.random.seed(1453)
W_CV = [[[np.random.uniform(-3, 3) for _ in range(n_feats)] for _ in range(n_folds)] for _ in range(n_models)]

# instantiate a cost hisory list containing one array for each model and fold
# each array corresponds to the most updated history of training costs for a given model
costsHistory_CV = [[[] for _ in range(n_folds)] for _ in range(n_models)]

# >> GRADIENT DESCENT
epoch = 0
tolerance = 0.005
modelsAndFoldsLeft = trainRdd_byModelAndFold.groupByKey().count()
convergences = 0
start = time.time()
while (epoch < n_epochs and modelsAndFoldsLeft > 0):

    epoch_start = time.time()
    
    # step 1) compute predictions with the current weights and append them to the original rdd
    hypothesis_CV = trainRdd_byModelAndFold\
    .map(lambda d_f__Xy: mxl.append_hypothesis(d_f__Xy, W_CV))\
    .cache() # <<<<< important to cache!
    

    # step 2) compute the total cost for the new predictions for every model d and every fold f
    costsDict_CV = hypothesis_CV\
    .mapValues(lambda X_y_h: mxl.get_cost_upd(X_y_h[1:]))\
    .reduceByKey(lambda df_upd1, df_upd2: df_upd1 + df_upd2).sortByKey(True)\
    .map(lambda d_f__costsum: mxl.get_train_cost(d_f__costsum, W_CV, HyperParams, 
                                                 trainRdd_foldsSizes))\
    .collectAsMap()
    
    # append the last training costs to the history list
    for (d, f) in costsDict_CV:
        newCost = costsDict_CV[(d, f)]
        costsHistory_CV[d][f].extend([newCost])
        
        if (epoch == 0):
            continue
        
        lastCost = costsHistory_CV[d][f][-2]
    
        # when the improvement in the cost is too small for a given model and fold,
        # keep only the rows that are not associated with it
        if (abs(lastCost - newCost) <= tolerance):
            trainRdd_byModelAndFold = trainRdd_byModelAndFold\
            .filter(lambda d_f__Xy: d_f__Xy[0] != (d, f))\
            .cache()
            
            modelsAndFoldsLeft -= 1
            print(">> Model {d}, fold {f} converged.".format(d=d, f=f))
        
        
    # step 3) 
    # - for every model and fold vector, create 1 new for every feature
    # - compute all the weights updates simoultaneously for all the models, folds and features
    # - get the updates in the form of a dictionary
    Wupds_CV = hypothesis_CV\
    .flatMap(lambda d_f__X_y_h: mxl.expand_features(d_f__X_y_h, n_feats))\
    .reduceByKey(lambda dfj_upd1, dfj_upd2: dfj_upd1 + dfj_upd2).sortByKey(True)\
    .collectAsMap()
    
    # update the weights for every model, fold, and feature in the dictionary
    for (d, f, j) in Wupds_CV:
        alpha = HyperParams[d][0]
        lambdareg = HyperParams[d][1]
        W_CV[d][f][j] = W_CV[d][f][j] * (1 - alpha * lambdareg) - alpha * Wupds_CV[(d, f, j)]
        
    epoch_end = time.time()
    print(epoch, " (", epoch_end - epoch_start, "s)")
    
    epoch += 1
        
end = time.time()
print("Total Gradient-Descent Running Time: ", (end-start)/60, "mins")
        
#%% MODEL SELECTION 2 - VALIDATION

# 1) once obtained some weights from Cross Validation training use these weights 
#    with validation data to compute a (validation) cost for every fold and model
# 2) average the validation costs over the different folds for all the models
#    so to obtain a CV error for every model trained
# 3) finally,select the model with minimum avg CV error to be the best model (bM)

modelsCVCostsDict = validationRdd_byModelAndFold\
.map(lambda d_f__Xy: mxl.append_hypothesis(d_f__Xy, W_CV))\
.mapValues(lambda X_y_h: mxl.get_cost_upd(X_y_h[1:]))\
.reduceByKey(lambda df_upd1, df_upd2: df_upd1 + df_upd2).sortByKey(True)\
.map(lambda d_f__costsum: mxl.get_train_cost(d_f__costsum, W_CV, HyperParams, validRdd_foldsSizes))\
.map(lambda df_cost: (df_cost[0][0], df_cost[1]))\
.reduceByKey(lambda d_cost1, d_cost2: d_cost1+d_cost2)\
.mapValues(lambda validation_cost_sum: validation_cost_sum / n_folds)\
.collectAsMap()

bestModel = min(modelsCVCostsDict, key=modelsCVCostsDict.get)

end = time.time()
print("Total Cross-Validation Elapsed Time: ", (end-start)/60, "mins")
print("Cross Validation Errors: ", modelsCVCostsDict)
print("Best Model: ", bestModel, 
      "(alpha: ", HyperParams[bestModel][0],", lambda: ", HyperParams[bestModel][1], ")")

#%% TRAINING SELECTED MODEL

# 1) once a model is selected, use its hyperparameters to train a new model
#    using all the non-test data

m = nonTestRdd.count()
alpha_bM = HyperParams[bestModel][0]
lambdareg_bM = HyperParams[bestModel][1]
W_train = [np.random.uniform(-3,3) for _ in range(n_feats)]

for epoch in range(n_epochs):

    epoch_start = time.time()
    
    # STEP 1: compute predictions with the new weights 
    # and append them to the original rdd
    hypothesis_bM = nonTestRdd\
    .map(lambda X_y: X_y + (mxl.predict(W_train, X_y[0]), ))\
    .cache() # <<<<< important to cache!
    

    #SECOND 2: compute the total cost for the new predictions 
    # for every model d and every fold f
    trainCost_bM = - hypothesis_bM\
    .map(lambda X_y_h: mxl.get_cost_upd(X_y_h[1:]))\
    .reduce(lambda upd1, upd2: upd1 + upd2) / m + ( lambdareg_bM / (2*m) * np.sum(np.square(W_train[1:])))
    
    
    # STEP 3: 
    # - for every model and fold vector, create 1 new for every feature
    # - compute all the weights updates simoultaneously for all the models, folds and features
    # - get the updates in the form of a dictionary
    Wupds_train = hypothesis_bM\
    .flatMap(lambda X_y_h: [(j, mxl.get_weight_upd(X_y_h, j)) for j in range(n_feats)])\
    .reduceByKey(lambda upd1, upd2: upd1 + upd2).sortByKey(True)\
    .collectAsMap()
    
    for j in Wupds_train:
        W_train[j] = W_train[j] * (1 - alpha_bM * lambdareg_bM) - alpha_bM * Wupds_train[j]
        
    epoch_end = time.time()
    
    print(epoch)

#%% TESTING & SCORING

# 1) at the end of the gradient descent, use the trained weights with test data
#    N.B. Now the prediction must be 0 or 1 --> we can use different decision boundaries
#         for the sigmoid function
# 2) compute different sets of predictions for every decision boundary considered
# 3) compute true positives, false pos, false neg, true neg for every db
# 4) use tp, fp, fn, tn to compute accuracy, precision, recall and fallout for every db
# 5) use fallout and recall to plot a roc curve

decision_boundaries = np.linspace(0,1,101)

scores = testRdd\
.flatMap(lambda X_y: [(db, (X_y[1], int(mxl.predict(W_train, X_y[0]) > db))) 
                      for db in decision_boundaries])\
.mapValues(lambda y_h: [int(y_h[0] == 1 and y_h[1] == 1),
                        int(y_h[0] == 0 and y_h[1] == 1),
                        int(y_h[0] == 1 and y_h[1] == 0),
                        int(y_h[0] == 0 and y_h[1] == 0)])\
.reduceByKey(lambda counters1, counters2: [sum(x) for x in zip(counters1, counters2)])\
.mapValues(lambda tp_fp_fn_tn: [(tp_fp_fn_tn[0] + tp_fp_fn_tn[3]) / sum(tp_fp_fn_tn),
                             tp_fp_fn_tn[0] / (tp_fp_fn_tn[0] + tp_fp_fn_tn[1]),
                             tp_fp_fn_tn[0] / (tp_fp_fn_tn[0] + tp_fp_fn_tn[2]),
                             tp_fp_fn_tn[1] / (tp_fp_fn_tn[1] + tp_fp_fn_tn[3]) ])\
.collectAsMap()

fallouts = [scores[db][3] for db in sorted(scores)]
recalls = [scores[db][2] for db in sorted(scores)]

# ROC Curve
plt.plot(fallouts, recalls)
plt.plot(np.linspace(0,1,101), np.linspace(0,1,101)[::-1])
