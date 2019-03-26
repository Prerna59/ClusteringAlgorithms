import numpy as np
from copy import deepcopy
import datetime
import os,random,math,shutil

def load_dataset(fileLocation):
    delimiter = "\t"
    cols = 0
    with open(fileLocation, "r") as file:
        data = file.readlines()
        cols = len(data[0].strip().split(delimiter))
    feature_matrix = np.loadtxt(fileLocation, delimiter="\t", usecols = range(2,cols))
    return feature_matrix

def run_mapreduce(Streaming_jar_path, mapper_path,reducer_path,input_folder,output_folder):
    os.system("hadoop jar"+" "+Streaming_jar_path+" "+"-file"+" "+mapper_path+" "+"-mapper"+" "+mapper_path+" "+"-file"+" "+reducer_path+" "+"-reducer"+" "+reducer_path+" "+"-input"+" "+input_folder+" "+"-output"+" "+output_folder)


def read_file(filename):
    file = open(filename, 'r')
    data = file.read()
    file.close()
    return data


def get_centroid(centroid_filename):
    current_centroid = []
    cluster_info = read_file(centroid_filename)
    for item in cluster_info.strip().split("\n"):
        data = item.strip().split("\t")	
        print(data) 
        cluster_id, points = data
        pointsarr = points.strip().split(";")
        current_centroid.append(pointsarr)
    return current_centroid

#calculates euclidean distance between two multi-dimensional points represented as list of strings(float values)
def euclidean_dist(a,b):
    #calculate distance between lists only
    val_sum = 0
    length = len(a)
    for i in range (length):
        diff = (float)(a[i]) - (float)(b[i])
        square = (diff) * (diff)
        val_sum += square
    return math.sqrt(val_sum)

#Deleting input folder
def store_file_in_hdfs(input_folder,output_folder):
    os.system("hdfs dfs -rm -r"+" "+input_folder)
    #Copying latest data into input folder
    os.system("hdfs dfs -put /home/hadoop/Downloads/input"+" "+input_folder)
    #Delete output folder for that file
    os.system("hadoop dfs -rm -r "+" "+output_folder)


def copy_files_to_local(local_output_path):
    shutil.rmtree(local_output_path+"/output_clusters")
    os.system("hdfs dfs -get output_clusters"+" "+local_output_path)
    os.system("hdfs dfs -rm -r output_clusters")


def getSSE(current_centroid,prev_centroid):
    sse = 0
    for i in range(len(current_centroid)):
        sse += euclidean_dist(current_centroid[i],prev_centroid[i])
    return sse

Streaming_jar_path = "/home/hadoop/hadoop/share/hadoop/tools/lib/hadoop-streaming-2.6.4.jar"
mapper_path = "/home/hadoop/Downloads/mapper.py"
reducer_path = "/home/hadoop/Downloads/reducer.py"
input_folder = "input_clusters"
output_folder = "output_clusters"
centroid_filename = "/home/hadoop/hadoop/cluster/output_clusters/part-00000"
store_file_in_hdfs(input_folder,output_folder)

#path to the input dataset for which the clusters have to be generated
feature_matrix = load_dataset("/home/hadoop/Downloads/input/dataset.txt")
local_output_path = "/home/hadoop/hadoop/cluster"
#number of clusters to be generated
k = 5
current_centroid = random.sample(list(feature_matrix), k)
prev_centroid = deepcopy(current_centroid)
sse = 1000
#Copying data from local file to HDFS
cluster_id = 0

print("Current Centroid::",current_centroid)

with open(centroid_filename, 'w') as file:
    for item in current_centroid:
        s = ""
        s += str(cluster_id)
        s += '\t'
        for val in range(len(item)):
            if val == len(item)-1:
                s += str(float(item[val]))
            else:
                s += str(float(item[val]))+";"
        s += '\n'
        file.write(s)
        cluster_id += 1

iteration = 0

while sse > 0:
    #copy reducer output to local fs so that the mapper can read the updated clusters in each iteration
    run_mapreduce(Streaming_jar_path, mapper_path,reducer_path,input_folder,output_folder)
    copy_files_to_local(local_output_path)
    prev_centroid = deepcopy(current_centroid)
    current_centroid = get_centroid(centroid_filename)
    print("Current centroid after mapreduce::", current_centroid)
    sse = getSSE(current_centroid, prev_centroid)
    iteration += 1
    print("Iteration::",iteration,"::SSE::",sse)
