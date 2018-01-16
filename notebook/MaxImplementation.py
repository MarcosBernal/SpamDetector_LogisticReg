
# coding: utf-8
import numpy as np
import time, math
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
n_folds = 4     # k-fold iterations
n_feats = len(nonTestRdd.first()[0]) # number of features

#%% PARAMETERS
n_models = 5    # number of models to train
n_epochs = 1000  # number of gradient descent iterations

#%% MODEL DEFINITION (HYPERPARAMETERS)
min_alpha0 = 0.0001; max_alpha0 = 0.005
min_lambdareg = 0.0; max_lambdareg = 10

HyperParams = sorted([(np.random.uniform(low=min_alpha0, high=max_alpha0),
                       np.random.uniform(low=min_lambdareg, high=max_lambdareg)) 
                      for _ in range(n_models)])
# !!!! USE THIS IF YOU WANT TO USE THE PARAMS I USED IN RESULT SECTION!!!!
#HyperParams = [(0.00016986440023933417, 3.6334049595152527), 
#               (0.00199639261790698, 7.227933759934101), 
#               (0.0027934899536763013, 9.887308201398312), 
#               (0.0029176354763686308, 2.8277596301087202), 
#               (0.0046918501668023384, 1.9870050706805031)]

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
.flatMap(lambda f__Xy: [((m, f__Xy[0]), f__Xy[1]) for m in range(n_models)])\
.cache()

validationRdd_byModelAndFold = validationRdd_ByFold\
.flatMap(lambda f__Xy: [((m, f__Xy[0]), f__Xy[1]) for m in range(n_models)])\
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
tolerance = 1e-05
modelsAndFoldsLeft = trainRdd_byModelAndFold.groupByKey().count()
start = time.time()
while (epoch < n_epochs and modelsAndFoldsLeft > 0):

    epoch_start = time.time()
    
    # step 1) compute predictions with the current weights and append them to the original rdd
    hypothesis_CV = trainRdd_byModelAndFold\
    .map(lambda m_f__Xy: mxl.append_hypothesis(m_f__Xy, W_CV))\
    .cache() # <<<<< important to cache!
    

    # step 2) compute the total cost for the new predictions for every model m and every fold f
    costsDict_CV = hypothesis_CV\
    .mapValues(lambda X_y_h: mxl.get_cost_upd(X_y_h[1:]))\
    .reduceByKey(lambda mf_upd1, mf_upd2: mf_upd1 + mf_upd2).sortByKey(True)\
    .map(lambda m_f__costsum: mxl.get_train_cost(m_f__costsum, W_CV, HyperParams, 
                                                 trainRdd_foldsSizes))\
    .collectAsMap()
    
    # append the last training costs to the history list
    for (m, f) in costsDict_CV:
        newCost = costsDict_CV[(m, f)]
        costsHistory_CV[m][f].extend([newCost])
        
        if (epoch == 0):
            continue
        
        lastCost = costsHistory_CV[m][f][-2]
    
        # when the improvement in the cost is too small for a given model and fold,
        # keep only the rows that are not associated with it
        if (abs(lastCost - newCost) <= tolerance):
            trainRdd_byModelAndFold = trainRdd_byModelAndFold\
            .filter(lambda m_f__Xy: m_f__Xy[0] != (m, f))\
            .cache()
            
            modelsAndFoldsLeft -= 1
            print(">> Model {m}, fold {f} converged.".format(m=m, f=f))
        
        
    # step 3) 
    # - for every model and fold vector, create 1 new for every feature
    # - compute all the weights updates simoultaneously for all the models, folds and features
    # - get the updates in the form of a dictionary
    Wupds_CV = hypothesis_CV\
    .flatMap(lambda m_f__X_y_h: mxl.expand_features(m_f__X_y_h, n_feats))\
    .reduceByKey(lambda mfj_upd1, mfj_upd2: mfj_upd1 + mfj_upd2).sortByKey(True)\
    .collectAsMap()
    
    # update the weights for every model, fold, and feature in the dictionary
    for (m, f, j) in Wupds_CV:
        alpha = HyperParams[m][0]
        lambdareg = HyperParams[m][1]
        W_CV[m][f][j] = W_CV[m][f][j] * (1 - alpha * lambdareg) - alpha * Wupds_CV[(m, f, j)]
        
    epoch_end = time.time()
    print(epoch, " (", epoch_end - epoch_start, "s)")
    
    epoch += 1
        
end = time.time()
print("Total Gradient-Descent Running Time: {mins}mins".format(mins=(end-start)/60))

# print for every model and fold the final gradient descent error and the number of epochs 
for m in range(len(costsHistory_CV)):
    for f in range(len(costsHistory_CV[0])):
        print("Model {m}, fold {f}: {err} ({ep})".format(m=m, f=f, err=costsHistory_CV[m][f][-1], ep=len(costsHistory_CV[m][f])))
       
#%% MODEL SELECTION 2 - VALIDATION

# 1) once obtained some weights from Cross Validation training use these weights 
#    with validation data to compute a (validation) cost for every fold and model
# 2) average the validation costs over the different folds for all the models
#    so to obtain a CV error for every model trained
# 3) finally,select the model with minimum avg CV error to be the best model (bM)

modelsCVCostsDict = validationRdd_byModelAndFold\
.map(lambda m_f__Xy: mxl.append_hypothesis(m_f__Xy, W_CV))\
.mapValues(lambda X_y_h: mxl.get_cost_upd(X_y_h[1:]))\
.reduceByKey(lambda mf_upd1, mf_upd2: mf_upd1 + mf_upd2).sortByKey(True)\
.map(lambda m_f__costsum: mxl.get_train_cost(m_f__costsum, W_CV, HyperParams, validRdd_foldsSizes))\
.map(lambda mf_cost: (mf_cost[0][0], mf_cost[1]))\
.reduceByKey(lambda m_cost1, m_cost2: m_cost1+m_cost2)\
.mapValues(lambda validation_cost_sum: validation_cost_sum / n_folds)\
.collectAsMap()

bestModel = min(modelsCVCostsDict, key=modelsCVCostsDict.get)

end = time.time()
print("Total Cross-Validation Elapsed Time: {mins}mins".format(mins=(end-start)/60))
print("Cross Validation Errors: ", modelsCVCostsDict)
print("Best Model: ", bestModel, 
      "(alpha: ", HyperParams[bestModel][0],", lambda: ", HyperParams[bestModel][1], ")")

#%% TRAINING SELECTED MODEL

# 1) once a model is selected, use its hyperparameters to train a new model
#    using all the non-test data

size = nonTestRdd.count()
alpha_bM = HyperParams[bestModel][0]
lambdareg_bM = HyperParams[bestModel][1]
W_train = [np.random.uniform(-3,3) for _ in range(n_feats)]

epoch_start = time.time()

epoch = 0
converged = False
previousCost = 99999
while (epoch < n_epochs and not converged):
    
    # STEP 1: compute predictions with the new weights 
    # and append them to the original rdd
    hypothesis_bM = nonTestRdd\
    .map(lambda X_y: X_y + (mxl.predict(W_train, X_y[0]), ))\
    .cache() # <<<<< important to cache!
    

    #SECOND 2: compute the total cost for the new predictions 
    trainCost_bM = - hypothesis_bM\
    .map(lambda X_y_h: mxl.get_cost_upd(X_y_h[1:]))\
    .reduce(lambda upd1, upd2: upd1 + upd2) / size + ( lambdareg_bM / (2*size) * np.sum(np.square(W_train[1:])))
    
    # check if the improvement of the cost is below the threshold
    if (abs(previousCost - trainCost_bM) <= tolerance):
        converged = True
    
    # update the last known cost
    previousCost = trainCost_bM
    
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
    epoch += 1

print("Total Training time: {mins}mins".format(mins=(epoch_end-epoch_start)/60))

#%% TESTING & SCORING

# 1) at the end of the gradient descent, use the trained weights with test data
#    N.B. Now the prediction must be 0 or 1 --> we can use different decision boundaries
#         for the sigmoid function
# 2) compute different sets of predictions for every decision boundary considered
# 3) compute true positives, false pos, false neg, true neg for every psi
# 4) use tp, fp, fn, tn to compute accuracy, precision, recall and fallout for every psi
# 5) use fallout and recall to plot a roc curve

Psi = np.linspace(0,1,501)[1:-1]

confusionMatrices = testRdd\
.flatMap(lambda X_y: [(psi, (X_y[1], int(mxl.predict(W_train, X_y[0]) > psi))) for psi in Psi])\
.mapValues(lambda y_h: [int(y_h == (1,1)), int(y_h == (0, 1)), int(y_h == (1, 0)), int(y_h == (0, 0))])\
.reduceByKey(lambda counters1, counters2: [sum(x) for x in zip(counters1, counters2)])

confusionMatrices_dict = confusionMatrices.collectAsMap()

advMetrics = confusionMatrices\
.mapValues(lambda tp_fp_fn_tn: [(tp_fp_fn_tn[0] + tp_fp_fn_tn[3]) / sum(tp_fp_fn_tn),
                             mxl.safe_division(tp_fp_fn_tn[0] , (tp_fp_fn_tn[0] + tp_fp_fn_tn[1])),
                             mxl.safe_division(tp_fp_fn_tn[0] , (tp_fp_fn_tn[0] + tp_fp_fn_tn[2])),
                             mxl.safe_division(tp_fp_fn_tn[1] , (tp_fp_fn_tn[1] + tp_fp_fn_tn[3])) ])\
.collectAsMap()

fallouts = [advMetrics[psi][3] for psi in advMetrics]
recalls = [advMetrics[psi][2] for psi in advMetrics]
f_r = [((advMetrics[psi][3], advMetrics[psi][2]), psi) for psi in advMetrics]
diagonal = sorted([(f, f) for f in fallouts])
inverse_diagonal = sorted([(f, 1-f) for f in fallouts])

# measure the delta y for the points of roc curve and the points of the inverse diagonal line
# the x at which the delta y is minimal it's a good estimator of where the two lines intersect;
# the psi of the point of intersection can be then selected as best psi
invDiagDist_x_y__psi = [(abs(y1 - y2), (x1, y1), psi) 
                    for ((x1,y1), psi), (x2, y2) in zip(sorted(f_r), sorted(inverse_diagonal))]
invDiagDist_bestResult = min(invDiagDist_x_y__psi)[1:]
print("----------------------------     FINAL MODEL BUNDARIES    ----------------------")
print(">> Best decision boundary (inv. diag. intersection): {best_psi} \n(Precision: {precision}, Accuracy: {accuracy})"\
      .format(best_psi=invDiagDist_bestResult[1], 
              accuracy=advMetrics[invDiagDist_bestResult[1]][0], 
              precision=advMetrics[invDiagDist_bestResult[1]][1]))
print("+++++++ Recall and Fallout metrics: \n(recall: {recall}, fallout: {fallout})"\
      .format(recall=advMetrics[invDiagDist_bestResult[1]][2], 
              fallout=advMetrics[invDiagDist_bestResult[1]][3]))
print("--------------------------------------------------------------------------")
# measure the distance from the non-discrimination line
# the psi of the point with max distance from the non-discrimination line can be 
# considered as the best psi
diagDist_x_y__psi = [(math.sqrt((x1 - x2)**2 + (y1 - y2)**2), (x1, y1), psi) 
                    for ((x1,y1), psi), (x2, y2) in zip(sorted(f_r), sorted(diagonal))]
diagDist_bestResult = max(diagDist_x_y__psi)[1:]
print(">> Best decision boundary (diag. distance): {best_psi} \n(Precision: {precision}, Accuracy: {accuracy})"\
      .format(best_psi=diagDist_bestResult[1], 
              accuracy=advMetrics[diagDist_bestResult[1]][0], 
              precision=advMetrics[diagDist_bestResult[1]][1]))
print("+++++++ Recall and Fallout metrics:  \n(recall: {recall}, fallout: {fallout})"\
      .format(recall=advMetrics[diagDist_bestResult[1]][2], 
              fallout=advMetrics[diagDist_bestResult[1]][3]))
print("--------------------------------------------------------------------------")

# measure the distance from the optimal point (0,1) for every point of the roc curve
# the point with the minimal distance is also the point with best psi
bestPointDist_x_y__psi = [(math.sqrt((0-x1)**2 + (1-y1)**2), (x1, y1), psi) 
                      for ((x1, y1), psi) in sorted(f_r)]
bestPointDist_bestResult = min(bestPointDist_x_y__psi)[1:]
print(">> Best decision boundary (best point distance): {best_psi} \n(Precision: {precision}, Accuracy: {accuracy})"\
      .format(best_psi=bestPointDist_bestResult[1], 
              accuracy=advMetrics[bestPointDist_bestResult[1]][0], 
              precision=advMetrics[bestPointDist_bestResult[1]][1]))
print("+++++++ Recall and Fallout metrics:  \n(recall: {recall}, fallout: {fallout})"\
      .format(recall=advMetrics[bestPointDist_bestResult[1]][2], 
              fallout=advMetrics[bestPointDist_bestResult[1]][3]))
#%%

# ROC Curve
fig, ax = plt.subplots(figsize=(10,8))
ax.set_ylabel('Recall', fontsize=17)
ax.set_xlabel('Fallout', fontsize=17)
ax.plot(sorted(fallouts), sorted(recalls)) # plot ROC curve
ax.plot([0, 1], [0, 1], "r-") # plot the non-discrimination line
ax.plot([0, 1], [1, 0], "r--") # plot the inverse diagonal
ax.scatter(invDiagDist_bestResult[0][0], invDiagDist_bestResult[0][1], s=30)
for (f,r), psi in f_r:
    if (np.random.rand() < 0.05):
        ax.annotate(str(round(psi, 3)), (f,r))
ax.set_xlim([0,1])
ax.set_ylim([0,1])
fig.savefig('ROC.png', dpi=100)
