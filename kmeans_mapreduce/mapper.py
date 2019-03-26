#!/usr/bin/env python
import sys,re,math
import numpy as np


def read_file(filename):
    file = open(filename, 'r')
    data = file.read()
    file.close()
    return data
    
def get_clusters(filename):
    cluster_info = read_file(filename)
    for item in cluster_info.strip().split("\n"):
        data = item.strip().split("\t")
        cluster_id, points = data
        pointsarr = points.strip().split(";")
        clusters.append((cluster_id, pointsarr))
        arrlength = len(pointsarr)
        initial_arr = [0] * arrlength
        clusters_gen_info[cluster_id] = (initial_arr,0)

def euclidean_dist(a,b):
    #calculate distance between lists only  
    val_sum = 0
    length = len(a)
    for i in range (length):
        diff = (float)(a[i]) - (float)(b[i])
        square = (diff) * (diff)
        val_sum += square
    return math.sqrt(val_sum)

def assign_cluster(point):
    #point  convert to np array ??
    closest_cluster_id = None
    min_dist = 1000
    for cluster in clusters:
        curr_dist = euclidean_dist(point,cluster[1])
        if curr_dist < min_dist:
            closest_cluster_id = cluster[0]
            min_dist = curr_dist
    return closest_cluster_id


filename = '/home/hadoop/hadoop/cluster/output_clusters/part-00000'

clusters = []
clusters_gen_info = dict()
get_clusters(filename)

regex = re.compile("\s+")
    
for item in sys.stdin:
    item = item.strip()
    point = regex.split(item)
    #if words == None:
        #print "Error parsing line -",line
        #continue;
    point_vals = []
    for i in range(2,len(point)):
        point_vals.append(float(point[i]))
    closest_cluster_id = assign_cluster(point_vals)
    sum_arr , count = clusters_gen_info[closest_cluster_id]
    sum_arr = [x + y for x, y in zip(sum_arr, point_vals)] # add two arrays ??
    clusters_gen_info[closest_cluster_id] = (sum_arr, count+1)
        
for key in clusters_gen_info:
    sum_arr, count = clusters_gen_info[key]
    #create ; separated string for sum_arr
    s = ""
    for i in range(len(sum_arr)):
        if i==len(sum_arr)-1:
            s+=str(sum_arr[i])
        else:
            s+=str(sum_arr[i])+";"
            
    print(key + "\t" + s + ";" + str(count))  