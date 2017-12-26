
# coding: utf-8
import numpy as np
import time
from pyspark import SparkContext, SparkConf
from marcoslib import normalize_data
import maxlib as mxl
import matplotlib.pyplot as plt

    
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
nonTestRdd, testRdd = master_norm_rdd.randomSplit([0.8, 0.2])

# save test set in a file
#testRdd.saveAsTextFile('spam.test.set')

########## PARALLELIZED CROSS-VALIDATION #####################################
#%% CONSTANTS
n_models = 3   # number of models to train
n_folds = 4     # k-fold iterations
n_epochs = 250  # number of gradient descent iterations
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

#%% CROSS VALIDATION SPLIT

# instantiate a Weights list containing one array for each model
# each array corresponds to the most updated weights vector for a given model
np.random.seed(1453)
W = [[[np.random.uniform(-3, 3) for _ in range(n_feats)] for _ in range(n_folds)] for _ in range(n_models)]

# instantiate a cost hisotry list containing one array for each model
# each array corresponds to the most updated history of training costs for a given model
trainCosts_history = [[[] for _ in range(n_folds)] for _ in range(n_models)]

  
#%% PLOTTING

#figs_grids = [ plt.subplots(nrows=2, ncols=2) for _ in range(n_models) ]
#plots = [[[] for _ in range(n_folds)] for _ in range(n_models)]
#for model, (fig, grid) in enumerate(figs_grids):
#    fig.suptitle("Model " + str(model) 
#                 + "\n(alpha: " + str(HyperParams[model][0])
#                 + ", lambda: " + str(HyperParams[model][1]) + ")")
#    for j, row in enumerate(grid):
#        for k, ax in enumerate(row):
#            fold = 2*j+k
#            ax.set_title("Fold " + str(fold+1))
#            ax.set_xlim([1, n_epochs])
#            ax.set_ylim([0,2])
#            line, = ax.plot([1])
#            plots[model][fold] = line
#
#plt.ion()
#plt.show()
#%% TRAINING (GRADIENT DESCENT)

start = time.time()
for epoch in range(n_epochs):

    epoch_start = time.time()
    
    # STEP 1: compute predictions with the new weights 
    # and append them to the original rdd
    hypothesis_rdd = trainRdd_byModelAndFold\
    .map(lambda d_f__Xy: mxl.append_hypothesis(d_f__Xy, W))\
    .cache() # <<<<< important to cache!
    

    #SECOND 2: compute the total cost for the new predictions 
    # for every model d and every fold f
    trainCosts_dict = hypothesis_rdd\
    .mapValues(lambda X_y_h: mxl.get_cost_upd(X_y_h[1:]))\
    .reduceByKey(lambda df_upd1, df_upd2: df_upd1 + df_upd2).sortByKey(True)\
    .map(lambda d_f__costsum: mxl.get_train_cost(d_f__costsum, W, HyperParams, 
                                                 trainRdd_foldsSizes))\
    .collectAsMap()
    
    # append the last training costs to the history list
    for (d, f) in trainCosts_dict:
        trainCosts_history[d][f].extend([trainCosts_dict[(d, f)]])
        #plots[d][f].set_ydata(trainCosts_history[d][f])
        #plots[d][f].set_xdata(range(1, epoch+2))
        
    #plt.pause(5)
    
    # STEP 3: 
    # - for every model and fold vector, create 1 new for every feature
    # - compute all the weights updates simoultaneously for all the models, folds and features
    # - get the updates in the form of a dictionary
    Wupds = hypothesis_rdd\
    .flatMap(lambda d_f__X_y_h: mxl.expand_features(d_f__X_y_h, n_feats))\
    .reduceByKey(lambda dfj_upd1, dfj_upd2: dfj_upd1 + dfj_upd2).sortByKey(True)\
    .collectAsMap()
    
    # update the weights for every model, fold, and feature in the dictionary
    for (d, f, j) in Wupds:
        alpha = HyperParams[d][0]
        lambdareg = HyperParams[d][1]
        #m = trainRdd_foldsSizes[f]
        W[d][f][j] = W[d][f][j] * (1 - alpha * lambdareg) - alpha * Wupds[(d, f, j)]
        
    epoch_end = time.time()
    
    print(epoch, " (", epoch_end - epoch_start, "s)")

        
end = time.time()
print("Total Gradient-Descent Running Time: ", (end-start)/60, "mins")
        
#%% VALIDATION

kfoldModelsCosts_dict = validationRdd_byModelAndFold\
.map(lambda d_f__Xy: mxl.append_hypothesis(d_f__Xy, W))\
.mapValues(lambda X_y_h: mxl.get_cost_upd(X_y_h[1:]))\
.reduceByKey(lambda df_upd1, df_upd2: df_upd1 + df_upd2).sortByKey(True)\
.map(lambda d_f__costsum: mxl.get_train_cost(d_f__costsum, W, HyperParams, validRdd_foldsSizes))\
.map(lambda df_cost: (df_cost[0][0], df_cost[1]))\
.reduceByKey(lambda d_cost1, d_cost2: d_cost1+d_cost2)\
.mapValues(lambda validation_cost_sum: validation_cost_sum / n_folds)\
.collectAsMap()

end = time.time()
best_model = min(kfoldModelsCosts_dict, key=kfoldModelsCosts_dict.get)

print("Total Cross-Validation Elapsed Time: ", (end-start)/60, "mins")
print("Cross Validation Errors: ", kfoldModelsCosts_dict)
print("Best Model: ", best_model, 
      "(alpha: ", HyperParams[best_model][0],", lambda: ", HyperParams[best_model][1], ")")

#%% TRAINING SELECTED MODEL

m = nonTestRdd.count()
alpha = HyperParams[best_model][0]
lambdareg = HyperParams[best_model][1]
W_bestModel = [np.random.uniform(-3,3) for _ in range(n_feats)]

for epoch in range(n_epochs):

    epoch_start = time.time()
    
    # STEP 1: compute predictions with the new weights 
    # and append them to the original rdd
    hypothesis_bestModel = nonTestRdd\
    .map(lambda X_y: X_y + (mxl.predict(W_bestModel, X_y[0]), ))\
    .cache() # <<<<< important to cache!
    

    #SECOND 2: compute the total cost for the new predictions 
    # for every model d and every fold f
    trainCost_bestModel = - hypothesis_bestModel\
    .map(lambda X_y_h: mxl.get_cost_upd(X_y_h[1:]))\
    .reduce(lambda upd1, upd2: upd1 + upd2) / m 
    + ( lambdareg / (2*m) * np.sum(np.square(W_bestModel[1:])))
    
    
    # STEP 3: 
    # - for every model and fold vector, create 1 new for every feature
    # - compute all the weights updates simoultaneously for all the models, folds and features
    # - get the updates in the form of a dictionary
    Wupds_bestModel = hypothesis_bestModel\
    .flatMap(lambda X_y_h: [(j, mxl.get_weight_upd(X_y_h, j)) for j in range(n_feats)])\
    .reduceByKey(lambda upd1, upd2: upd1 + upd2).sortByKey(True)\
    .collectAsMap()
    
    for j in Wupds_bestModel:
        W_bestModel[j] = W_bestModel[j] * (1 - alpha * lambdareg) - alpha * Wupds_bestModel[j]
        
    epoch_end = time.time()
    
    print(epoch)

#%% TESTING

decision_boundaries = np.linspace(0,1,101)

# > create a new rdd for every decision boundary (db)
# > compute the number of true positives, false pos, false neg, true neg
# > use these values to compute accuracy, precision, recall and fallout per each db
scores = testRdd\
.flatMap(lambda X_y: [(db, (X_y[1], 0.0 if mxl.predict(W_bestModel, X_y[0]) <= db else 1.0)) 
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

roc_x = [scores[k][3] for k in sorted(scores)][::-1]
roc_y = [scores[k][2] for k in sorted(scores)][::-1]

# ROC Curve
plt.plot(roc_x, roc_y)
plt.plot(np.linspace(0,1,101), np.linspace(0,1,101)[::-1])
