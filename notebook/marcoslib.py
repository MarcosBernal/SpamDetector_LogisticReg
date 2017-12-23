#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 12:37:42 2017

@author: -
"""

# Normalize values
def normalize_data(data):
    max_min = data.flatMap(lambda x: [ (index_key, x[index_key]) for index_key in range(len(x)-1)]) #Last position is label
    max__list = sorted(max_min.reduceByKey(lambda x,y: x if x > y else y).collect())
    min__list = sorted(max_min.reduceByKey(lambda x,y: x if x < y else y).collect())
    mean_list = sorted([ value[1]/data.count() for value in max_min.reduceByKey(lambda x,y: x + y).collect()])
    
    return data.map(lambda x: [(float(x[index]) - min__list[index][1])/(max__list[index][1] - min__list[index][1]) if index != len(x)-1 else x[index] for index in range(len(x))] )   
