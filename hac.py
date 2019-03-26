import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.spatial import distance
from copy import deepcopy
from scipy.misc import comb
import itertools
import sys
def load_dataset(fileLocation):
    delimiter = "\t"
    cols = 0
    with open(fileLocation, "r") as file:
        data = file.readlines()
        cols = len(data[0].strip().split(delimiter))
    actual_clusters = np.loadtxt(fileLocation, delimiter="\t", usecols = [1])
    feature_matrix = np.loadtxt(fileLocation, delimiter="\t", usecols = range(2,cols))
    mean_feature_matrix = feature_matrix.mean(axis = 0)
    feature_matrix = np.loadtxt(fileLocation, delimiter="\t", usecols = range(2,cols))
    deviation = np.std(feature_matrix)
    feature_matrix = (feature_matrix - feature_matrix.mean(axis = 0, keepdims = True)) / (deviation)
    return actual_clusters, feature_matrix

def euclidean_dist(a,b,ax = 1):
    return np.linalg.norm(a-b, axis = ax)


def generate_mapped_clusters_hac(clusters_dict,data_len):
    cluster_id = 1
    mapped_clusters = np.zeros(data_len, dtype = np.float32)
    for item in clusters_dict:
        temp_list = item[1:len(item)-2].split(",")
        for val in temp_list:
            index = int(val.strip())
            mapped_clusters[index] = cluster_id
        cluster_id += 1
    return mapped_clusters

#HAC implementation
def hac(feature_matrix):
    data_len = feature_matrix.shape[0]
    distance_matrix = np.zeros((data_len, data_len))
    initial_clusters_list = [[i] for i in range(0, data_len)]
    clusters_list = []
    clusters_list.append(initial_clusters_list)
    cluster_set = set()
    orig_distance_matrix = np.zeros((data_len, data_len))
    max_dist_matrix = 10000
    for i in range(0, data_len):
        cluster_set.add(i)
    for i in range (data_len):
        for j in range (i, data_len):
            if i==j:
                distance_matrix[i][j] = max_dist_matrix
                distance_matrix[j][i] = max_dist_matrix
                orig_distance_matrix[i][j] = max_dist_matrix
                orig_distance_matrix[j][i] = max_dist_matrix
                continue;
            dist = euclidean_dist(feature_matrix[i][:],feature_matrix[j][:],None)
            distance_matrix[i][j] = dist
            distance_matrix[j][i] = dist
            orig_distance_matrix[i][j] = dist
            orig_distance_matrix[j][i] = dist
    cluster_map = dict()
    while len(cluster_set) != 10:
        #find minimum value indices ->
        min_val = np.min(distance_matrix)
        min_row = 0
        min_col = 0
        flag = False
        for i in range(data_len):
            for j in range(data_len):
                if(distance_matrix[i][j] == min_val):
                    min_row = i
                    min_col = j
                    flag = True
                    break;
            if flag:
                break;
        del_val = max(min_row, min_col)
        #if max val is present in cluster map -> delete that as the key and add this
        #val as the key and add in the cluster
        #remove the larger val from cluster set
        cluster_set.discard(del_val)
        #keep the lower val as the val where we still store the updated indices
        store_val = min(min_row, min_col)
        store_key = str(store_val)
        if str(del_val) in cluster_map:
            cluster_list = cluster_map[str(del_val)]
            #store_val might have its own cluster already created -> merge these two
            second_list = []
            if str(store_val) in cluster_map:
                second_list = cluster_map[str(store_val)]
            else:
                second_list = [store_val]
            merged_list = cluster_list + second_list
            del cluster_map[str(del_val)]
            cluster_map[str(store_val)] = merged_list
        elif str(store_val) in cluster_map:
            cluster_list = cluster_map[str(store_val)]
            cluster_list.append(del_val)
            cluster_map[str(store_val)] = cluster_list
        else:
            cluster_map[str(store_val)] = [min_row, min_col]
        updated_cluster = cluster_map[store_key]
        temp_list = []
        for i in range(data_len):
            distance_matrix[i][del_val] = max_dist_matrix
            distance_matrix[del_val][i] = max_dist_matrix
        min_index = min(updated_cluster)
        for val in cluster_set:
            temp_cluster_list = []
            if val == store_val:
                if str(val) in cluster_map:
                    curr_cluster = cluster_map[str(val)]
                    temp_cluster_list = deepcopy(curr_cluster)
                else:
                    temp_cluster_list = [val]
                temp_list.append(temp_cluster_list)
                continue
            #current matrix value ->
            min_dist = max_dist_matrix
            #val in cluster set can be a previous cluster
            temp_cluster_list = []
            if str(val) in cluster_map:
                curr_cluster = cluster_map[str(val)]
                temp_cluster_list = deepcopy(curr_cluster)
                for i in curr_cluster:
                    for j in updated_cluster:
                        curr_dist = orig_distance_matrix[i][j]
                        if(curr_dist < min_dist):
                            min_dist = curr_dist
                min_index_curr = min(curr_cluster)
                distance_matrix[min_index][min_index_curr] = min_dist
                distance_matrix[min_index_curr][min_index] = min_dist
            else:
                for index in updated_cluster:
                    curr_dist = orig_distance_matrix[val][index]
                    if(curr_dist < min_dist):
                        min_dist = curr_dist
                    #fill the distance matrix for this val
                distance_matrix[min_index][val] = min_dist
                distance_matrix[val][min_index] = min_dist
                temp_cluster_list = [val]
            temp_list.append(temp_cluster_list)
        clusters_list.append(temp_list)
    temp_dict = dict()
    temp_list = clusters_list[len(clusters_list)-1]
    for temp in temp_list:
        if str(temp) in temp_dict:
            temp_dict[str(temp)] += 1
        else:
            temp_dict[str(temp)] = 1
    return generate_mapped_clusters_hac(temp_dict, data_len)

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

# fileName = "iyer.txt"
# actual_clusters, feature_matrix = load_dataset("/Users/prernasingh/Documents/DataMining/Project2/iyer.txt")
fileName = input("Enter name of file: ")
feature_matrix, actual_matrix = load_dataset(fileName)  
mapped_clusters = hac(feature_matrix)
fileName = "iyer.txt"
print ("Rand index: ", rand_index(actual_clusters, mapped_clusters))
pca_mat = PCA(feature_matrix)
unique_cluster_number = set(mapped_clusters)
unique_cluster = {}
for n in range(0, len(unique_cluster_number)):
    unique_cluster[n] = n
plot_graph(fileName, mapped_clusters, unique_cluster, unique_cluster_number, pca_mat)
