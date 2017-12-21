
# coding: utf-8

# In[2]:


import math, time, random
import numpy as np
from random import shuffle
from pyspark import SparkContext, SparkConf, rdd


# In[4]:


# Normalize values
def normalize_data(data):
    max_min = data.flatMap(lambda x: [ (index_key, x[index_key]) for index_key in range(len(x)-1)]) #Last position is label
    max__list = sorted(max_min.reduceByKey(lambda x,y: x if x > y else y).collect())
    min__list = sorted(max_min.reduceByKey(lambda x,y: x if x < y else y).collect())
    mean_list = sorted([ value[1]/data.count() for value in max_min.reduceByKey(lambda x,y: x + y).collect()])
    
    return data.map(lambda x: [(float(x[index]) - min__list[index][1])/(max__list[index][1] - min__list[index][1]) if index != len(x)-1 else x[index] for index in range(len(x))] )   


# In[25]:


# given a tuple of sub-rdds and the cross-validation iteration index,
#  this method returns a tuple containing training and validation rdds
def get_train_validation_rdds(sub_rdds, k, indices=list(range(0, 4))):
    
    # the validation set is the k-th sub-rdd
    validation_rdd = sub_rdds[indices.pop(k)]
    
    # initialize the train rdd with the first sub-rdd left
    train_rdd = sub_rdds[indices.pop(0)]
    
    # append all the remaining sub-rdds to the train-rdd
    for i in indices:
        train_rdd = train_rdd.union(sub_rdds[i])
    
    # save train and validation set in a file
    validation_rdd.saveAsTextFile('spam.validation' + str(k+1) + '.norm.data')
    train_rdd.saveAsTextFile('spam.train' + str(k+1) + '.data')
    
    return train_rdd, validation_rdd


# In[6]:


# x is the features vector without label
# w is the weights vector
# b is the bias
def predict(w, x, b):
    return (1 / (1 + math.exp(-(np.dot(w, x)+b))))

def get_cost_upd(x_y_yhat):
    x, y, yhat = x_y_yhat
    return y * math.log(yhat) + (1-y) * math.log(1-yhat)

def get_weight_upd(x_y_yhat, j):
    x, y, yhat = x_y_yhat
    return (yhat - y) * x[j]


# In[7]:


conf = SparkConf().setAppName("Spam Filter").setMaster("local[1]").set("spark.hadoop.validateOutputSpecs", "false");
sc = SparkContext(conf=conf)

file_object  = open('spam.data', 'r')
lines = file_object.readlines()
file_object.close()
    
total_size = len(lines)


# In[19]:


# Creating RDD
master_rdd = sc.parallelize(lines)
master_rdd = master_rdd.map(lambda x: [float(item) for item in x.split('\n')[0].split(" ")])
master_norm_rdd = normalize_data(master_rdd)
master_norm_rdd = master_norm_rdd.map(lambda x: (x[:-1], x[-1]))


# In[20]:


print(master_norm_rdd.first())


# In[26]:


# divide the original rdd in non-test and test rdds
non_test_rdd, test_rdd = master_norm_rdd.randomSplit([0.8, 0.2])
non_test_rdd = non_test_rdd.cache()
test_rdd = test_rdd.cache()

# save test set in a file
test_rdd.saveAsTextFile('spam.test.set')

# divide the non-test rdd in 4 sub-rdds
sub_rdds = non_test_rdd.randomSplit([0.25, 0.25, 0.25, 0.25])

# k-fold iterations
for k in range(0, 4):
    # for every iteration get a different train and validation sets
    train_rdd, validation_rdd = get_train_validation_rdds(sub_rdds, k, indices=list(range(0,4)))


# In[29]:


X, y = train_rdd.first()
y


# In[41]:


train_rdd = train_rdd.cache()

# compute useful constants for further computations
m = train_rdd.count()
alpha = 1
lambdareg = 0 
learnrate = alpha/m

# initialize the true labels vector
n = len(train_rdd.first()[0]) 
print("#Features: ", n)

# initialize the weights vector (one weight per feature) and bias
new_weights = np.zeros(n)
new_bias = 0

import time
start = time.time()
for epoch in range(400):
    
    weights = new_weights
    bias = new_bias

    #FIRST STEP: compute the predictions for the given weights and append them to the rest
    # REMEMBER: every row of the rdd is now a tuple (feature_vector, true_label)
    xs_ys_yhats_rdd = train_rdd    .map(lambda x_y: x_y + (predict(weights, x_y[0], bias),))    .cache()

    #SECOND STEP: compute the total cost for the computed predictions
    cost = xs_ys_yhats_rdd    .map(lambda x_y_yhat: get_cost_upd(x_y_yhat))    .reduce(lambda c1, c2: c1+c2)

    # (regularization)
    cost_reg_term = lambdareg/(2*m) + sum([w**2 for w in weights])
    cost = -1/m * cost - cost_reg_term
    
    if (epoch % 50 == 0):
        print("(", epoch, ") Cost: ", cost)

    #THIRD STEP: update all the weights simoultaneously
    # 3.1. get the updating term for all the weights
    weights_upds = xs_ys_yhats_rdd    .flatMap(lambda x_y_yhat: [(j, get_weight_upd(x_y_yhat, j))
                               for j in range(n)])\
    .reduceByKey(lambda u1, u2: u1+u2)\
    .sortByKey(True)\
    .map(lambda j_weightsumupds: j_weightsumupds[1])\
    .collect()
    
    bias_upd = xs_ys_yhats_rdd    .map(lambda x_y_yhat: x_y_yhat[1] - x_y_yhat[2])    .reduce(lambda p, q: p+q)
    
    # 3.2. update the old weights (with regularization)
    weight_reg_term = (1 - learnrate * lambdareg)
    new_weights = [weight * weight_reg_term - learnrate * weight_upd 
                   for weight, weight_upd in zip(weights, weights_upds)]
    
    #new_bias = bias * weight_reg_term - alpha / m * bias_upd
    
end = time.time()
print()
print("Cost: ", cost)
print("Weights: ", weights)
print("> Total elapsed time: ", ((end-start)/60), "mins")


# In[85]:


sum([w**2 for w in weights])


# In[65]:




print(weights_upds)
#print(" 1st el (", type(f[0]), "): ", f[0])
#print(" 2nd el (", type(f[1]), "): ", f[1])
#print("Keys: ", len(set(weights_upds.keys().collect())))


# In[12]:



.reduceByKey(lambda xj_y_yhat1, xj_y_yhat2: 
             get_weight_upd(xj_y_yhat1, xj_y_yhat2))\
.map(lambda j_weightsupds: - l_rate_over_size * j_weightsupds[1]).collect()

new_weights = [sum(_) for _ in zip(weights, weights_upds)]


# In[49]:


len(train_rdd.first())


# In[32]:


x = (2,3,4)


# In[36]:


x + (3,)

