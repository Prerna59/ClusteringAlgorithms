#!/usr/bin/env python
from __future__ import division
import sys

#assuming that mappers give sorted output to reducers

prevKey = None
points_count_arr = []
total_count = 0

def emit_updated_clusters(cluster_id, points_count_arr, count):
    if count == 0:
        return
    else:
        points_count_arr = [x / count for x in points_count_arr]

    s = ""
    for i in range(len(points_count_arr)):
        if(i == len(points_count_arr)-1):
            s+=str(points_count_arr[i])
        else:
            s+=str(points_count_arr[i])+";"
    print(cluster_id + "\t" + s)

for item in sys.stdin:
    emitted_data = item.strip().split("\t")
    if len(emitted_data) != 2:
        continue;
    cluster_id, points_sum = emitted_data
    points_arr = points_sum.strip().split(";")
    count = (int)(points_arr[len(points_arr)-1])
    points_arr = points_arr[:len(points_arr)-1]
    points_val = []
    for i in range(len(points_arr)):
        points_val.append(float(points_arr[i]))
    if prevKey and prevKey != cluster_id:
        emit_updated_clusters(prevKey,points_count_arr,total_count)
        prevKey = None
        points_count_arr = []
        total_count = 0
        
    prevKey = cluster_id
    if not points_count_arr:
        points_count_arr = [0] * len(points_val)
    points_count_arr = [x + y for x, y in zip(points_count_arr, points_val)] 
    total_count += count
    
if prevKey != None:
    emit_updated_clusters(prevKey,points_count_arr,total_count)