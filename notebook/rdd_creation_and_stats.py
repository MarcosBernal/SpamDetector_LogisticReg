#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math
from pyspark import SparkContext, SparkConf

# If spark context was not defined before
n_executors_spark = 6
conf = SparkConf().setAppName("Spam Filter").setMaster("local["+str(n_executors_spark)+"]").set("spark.hadoop.validateOutputSpecs", "false");
sc =   SparkContext.getOrCreate(conf=conf)


file_object  = open('spam.data', 'r')
lines = file_object.readlines()
file_object.close()
    
total_size = len(lines)

# Creating RDD
master_rdd = sc.parallelize(lines)
master_rdd = master_rdd.map(lambda x: [float(item) for item in x.split('\n')[0].split(" ")])
master_rdd = master_rdd.cache() # Persist this RDD with the default storage level (MEMORY_ONLY).

# Get stats of the instance values
max_min_rdd = master_rdd.flatMap(lambda x: [ (index_key, x[index_key]) for index_key in range(len(x))])
max_list = [ item[1] for item in max_min_rdd.reduceByKey(lambda x,y: x if x > y else y).collect()]
min_list = [ item[1] for item in max_min_rdd.reduceByKey(lambda x,y: x if x < y else y).collect()]
mean_list = [ value[1]/len(lines) for value in max_min_rdd.reduceByKey(lambda x,y: x + y).collect()]
std_deviation_list = [ math.sqrt((value[1]/len(lines))) for value in max_min_rdd.map(lambda x:  [ x[0],(x[1] - mean_list[x[0]])*(x[1] - mean_list[x[0]]) ]).reduceByKey(lambda x,y: x+y).collect() ]
coeff_variation = [ std_deviation_list[index]/mean_list[index] for index in range(len(std_deviation_list))]

# Normalize values
def normalize_data(data):
    max_min = data.flatMap(lambda x: [ (index_key, x[index_key]) for index_key in range(len(x)-1)]) #Last position is label
    max__list = sorted(max_min.reduceByKey(lambda x,y: x if x > y else y).collect())
    min__list = sorted(max_min.reduceByKey(lambda x,y: x if x < y else y).collect())
    
    return data.map(lambda x: [(float(x[index]) - min__list[index][1])/(max__list[index][1] - min__list[index][1]) if index != len(x)-1 else x[index] for index in range(len(x))] )
    

    
master_norm_rdd = normalize_data(master_rdd).cache()
#.map(lambda x: [(x[index] - min_list[index])/(max_list[index] - min_list[index]) for index in range(len(x)-1)] )
max_min_norm_rdd = master_norm_rdd.flatMap(lambda x: [ (index_key, x[index_key]) for index_key in range(len(x))])

print("--")
print("Calculated stats of the features of all instances: max, min, mean, std_deviation, and coeff_variation")
print("Values have been normalized and are kept in the \'master_norm_rdd\' variable")
print("--")

train_validation_rdd, test_rdd = master_norm_rdd.randomSplit([0.8, 0.2])
train_validation_rdd = train_validation_rdd.cache()
test_rdd = test_rdd.cache()

# save test set in a file
# test_rdd.saveAsTextFile('spam.test.set')

print("Train & Validation: ", train_validation_rdd.count(), " samples")
print("Test: ", test_rdd.count(), " samples")
print("> Do sizes match? ", train_validation_rdd.count() + test_rdd.count() == total_size)
