import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.spatial import distance
from copy import deepcopy

#Loading dataset from local folder to project
def load_dataset(fileLocation):
    delimiter = "\t"
    cols = 0
    with open(fileLocation, "r") as file:
        data = file.readlines()
        cols = len(data[0].strip().split(delimiter))
    feature_matrix = np.loadtxt(fileLocation, delimiter="\t", usecols = range(2,cols))
    actual_matrix = np.loadtxt(fileLocation, delimiter="\t", usecols = [1])
    return feature_matrix, actual_matrix

#Calculating Euclidean distance
def distance(a,b,ax = 1):
    return np.linalg.norm(a-b, axis = ax)

def Kmeans(feature_matrix):
    #lets vary k from 1 to 15 and check the results
    k = int(input("Enter the value for k"))
    #k = 5
    initial_data = random.sample(list(feature_matrix), k)
    current_centroid = np.asarray(initial_data)
    centroid_shape = (current_centroid.shape[0], current_centroid.shape[1])
    prev_centroid = np.zeros(centroid_shape, dtype = np.float32)
    data_len = feature_matrix.shape[0]
    mapped_clusters = np.zeros(data_len, dtype = np.int32)
    sse = distance(current_centroid, prev_centroid, None)
    
    #keep computing until the centroids do not change
    iteration = 0
    while sse != 0 :
        for i in range(data_len):
            cluster_dist = distance(feature_matrix[i][:], current_centroid)
            cluster_index = np.argmin(cluster_dist)
            mapped_clusters[i] = cluster_index + 1
        prev_centroid = deepcopy(current_centroid)
        #update current centroids
        for val in range(k):
            cluster_points = [feature_matrix[j][:] for j in range(data_len) if mapped_clusters[j]-1==val]
            current_centroid[val] = np.mean(cluster_points, axis=0)
        iteration += 1
        sse = distance(current_centroid, prev_centroid, None)
        print("Iteration::",iteration,"SSE:",sse)
    return mapped_clusters


######## PCA code##############
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

############### Rand index code #############
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
# feature_matrix, actual_matrix = load_dataset("/Users/prernasingh/Documents/DataMining/Project2/iyer.txt")
fileName = input("Enter name of file: ")
feature_matrix, actual_matrix = load_dataset(fileName)
mapped_clusters = Kmeans(feature_matrix)
print ("Rand index: ", rand_index(actual_matrix, mapped_clusters))

#Calling PCA method to visualize the data
pca_mat = PCA(feature_matrix)
unique_cluster_number = set(mapped_clusters)
unique_cluster = {}
for n in range(0, len(unique_cluster_number)):
    unique_cluster[n] = n
plot_graph(fileName, mapped_clusters, unique_cluster, unique_cluster_number, pca_mat)
