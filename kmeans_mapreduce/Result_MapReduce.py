import numpy as np
import matplotlib.pyplot as plt
import random
from copy import deepcopy
import datetime

#read clusters from the updated file
#assign each point to its cluster

def read_file(filename):
    file = open(filename, 'r')
    data = file.read()
    file.close()
    del file
    return data

def load_dataset(fileLocation):
    col_num = -1
    delimiter = "\t"
    cols = 0
    with open(fileLocation, "r") as file:
        data = file.readlines()
        cols = len(data[0].strip().split(delimiter))

    actual_clusters = np.loadtxt(fileLocation, delimiter="\t", usecols = [1])
    feature_matrix = np.loadtxt(fileLocation, delimiter="\t", usecols = range(2,cols))
    mean_feature_matrix = feature_matrix.mean(axis = 0)
    feature_matrix = np.loadtxt(fileLocation, delimiter="\t", usecols = range(2,cols))
    #deviation = np.std(feature_matrix)
    #feature_matrix = (feature_matrix - feature_matrix.mean(axis = 0, keepdims = True)) / (deviation)
    return actual_clusters, feature_matrix

def euclidean_dist(a,b,ax = 1):
    return np.linalg.norm(a-b, axis = ax)

actual_clusters, feature_matrix = load_dataset("/Users/prernasingh/Documents/DataMining/Project2/iyer.txt")
final_clusters_file = "/Users/prernasingh/Documents/DataMining/Project2/Shivani_data/iyer/iyer-12-iterations-30"
cluster_info = read_file(final_clusters_file)
k = 12
final_centroid = np.empty((0,k), dtype = np.float32)
for item in cluster_info.strip().split("\n"):
    data = item.strip().split("\t")
    cluster_id, points = data
    pointsarr = points.strip().split(";")
    temp_list = []
    for val in pointsarr:
        temp_list.append(float(val))
    final_centroid = np.append(final_centroid, np.array([temp_list]),axis = 0)

data_len = feature_matrix.shape[0]
mapped_clusters = np.zeros(data_len, int)
print("Final centroid shape::",final_centroid.shape)
for i in range(data_len):
    cluster_dist = euclidean_dist(feature_matrix[i][:], final_centroid)
    cluster_index = np.argmin(cluster_dist)
    mapped_clusters[i] = cluster_index + 1
print(len(mapped_clusters))
#read points from dataset and find their final cluster
def PCA(feature_matrix):
    #Adjusted Matrix
    adjMatrix = feature_matrix - feature_matrix.mean(axis = 0) 
    #Covariance of adjusted matrix
    covMatrix = np.cov(adjMatrix.T)
    #Eigen value and Eigen vector
    eigenVal, eigenVec = np.linalg.eig(covMatrix)
    #Finding top n eigen value for principle component, In this case we are finding 2
    # -2 represents the top k values 
    index = eigenVal.argsort()[-2:][::-1]   
    eigenVal_top2 = eigenVal[index]
    eigenVec_top2 = eigenVec[:,index]
    #Need to build priniciple components from n dimensions to two dimensions
    row = feature_matrix.shape[0]
    col = eigenVec_top2.shape[1] # will be two for two dimensions
    pca_mat = np.empty([row,col])
    for i, j in enumerate(eigenVec_top2.T):
        pca_mat[:,i] = np.dot(adjMatrix, j)   
    return pca_mat

def plot_graph(file_name, clusters, unique_cluster, unique_cluster_number, feature_matrix):
    # to assign colors needed
    colors = []
    for i in unique_cluster_number:
        colors.append(plt.cm.jet(float(i) / max(unique_cluster_number)))
    i = 0
    for cluster in unique_cluster:
        x = []
        for j, p in enumerate(feature_matrix[:,0]):
            if clusters[j] == cluster:
                x.append(p)
        y = []
        for k, p in enumerate(feature_matrix[:,1]):
            if clusters[k] == cluster:
                y.append(p)
        plt.scatter(x, y, c=colors[i], label=str(cluster))
        i = i+1
    plt.title(file_name)
    plt.legend()
    plt.show()
    
def prepare_comparison(ground_truth_clusters, clusters):
    dimension = ground_truth_clusters.shape[0]
    cluster_matrix = np.zeros((dimension, dimension))
    ground_truth_matrix = np.zeros((dimension, dimension))

    for same in range(0, dimension):
        cluster_matrix[same][same] = 1
        ground_truth_matrix[same][same] = 1

    for row in range(0, dimension):
        for col in range(row + 1, dimension):
            if clusters[row] == clusters[col]:
                cluster_matrix[row][col] = 1
                cluster_matrix[col][row] = 1
            if ground_truth_clusters[row] == ground_truth_clusters[col]:
                if ground_truth_clusters[col] != -1:
                    ground_truth_matrix[row][col] = 1
                    ground_truth_matrix[col][row] = 1

    return cluster_matrix, ground_truth_matrix

def rand_index(ground_truth_clusters, clusters):
    cluster_matrix, ground_truth_matrix = prepare_comparison(ground_truth_clusters, clusters)

    one_and_one = 0.
    zero_and_zero = 0.
    one_or_one = 0. # one or one includes one and one

    for row in range(0, cluster_matrix.shape[0]):
        for col in range(0, cluster_matrix.shape[0]):
            if cluster_matrix[row][col] == 1 or ground_truth_matrix[row][col] == 1:
                one_or_one += 1
            if cluster_matrix[row][col] == 1 and ground_truth_matrix[row][col] == 1:
                one_and_one += 1
            if cluster_matrix[row][col] == 0 and ground_truth_matrix[row][col] == 0:
                zero_and_zero += 1

    return (one_and_one + zero_and_zero) / (one_or_one + zero_and_zero)
print ("Rand index: ", rand_index(actual_clusters, mapped_clusters))
fileName = "iyer.txt"
pca_mat = PCA(feature_matrix)
unique_cluster_number = set(mapped_clusters)
unique_cluster = {}
for n in range(0, len(unique_cluster_number)):
    unique_cluster[n] = n

plot_graph(fileName, mapped_clusters, unique_cluster, unique_cluster_number, pca_mat)