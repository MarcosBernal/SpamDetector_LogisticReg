
# coding: utf-8
import numpy as np
import time
from pyspark import SparkContext, SparkConf
from marcoslib import normalize_data
from maxlib import get_train_validation_rdds, gradient_descent, get_cost

    
n_executors_spark = 4
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
test_rdd.saveAsTextFile('spam.test.set')

##############################################################################

# k-fold iterations
n_folds = 4

# divide the non-test rdd in 4 sub-rdds
sub_rdds = non_test_rdd.randomSplit(np.repeat(1.0/n_folds, n_folds))


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

start = time.time()
kfold_cost = 0
for k in range(0, n_folds):
    iteration_start = time.time()
    print("=== Fold ", k+1, " ====================================")
    
    # for every iteration get a different train and validation sets
    train_rdd, validation_rdd = get_train_validation_rdds(sub_rdds, k)
    
    # train the model and get some weights
    train_cost, train_W = gradient_descent(train_rdd, n_epochs=400, alpha0=0.3)
    
    # validate the model and get the cost (to be averaged)
    validation_cost = get_cost(validation_rdd, train_W) 
    kfold_cost += validation_cost
    
    iteration_end = time.time()

    print("> Train/Validation Cost: ", train_cost, "/", validation_cost)
    print("> Total Elapsed time: ", (iteration_end-iteration_start)/60, "mins")
    print()
    
kfold_cost /= n_folds
end = time.time()

print("=== ", n_folds, "-fold Cross Validation =================")
print("> Avg cost: ", kfold_cost)
print("> Elapsed time: ", ((end-start)/60), "mins")




