# References: Clustering 4 slides - DBSCAN pseudocode, and other slides
import numpy as np
import matplotlib.pyplot as plt

# runs dbscan and returns a list of points with their assigned cluster
def dbscan(coordinates, epsilon, min_points):
    number_of_points = coordinates.shape[0]
    visited = np.zeros((number_of_points), dtype=bool) # track visited points
    clusters = np.zeros(number_of_points) # cluster assignments, 0 is noise
    C = 0

    for P in range(0, number_of_points):
        if visited[P] != True: # if not visited
            visited[P] = True
            neighbor_points = region_query(coordinates, P, epsilon)
            if len(neighbor_points) >= min_points: # if not noise
                C = C + 1
                expand_cluster(coordinates, P, neighbor_points, C,
                                epsilon, min_points, clusters, visited)

    return clusters

# expand clusters that meet min_point threshold
def expand_cluster(coordinates, P, neighbor_points, C,
                    epsilon, min_points, clusters, visited):
    clusters[P] = C # add P to cluster C
    for each_point in neighbor_points:
        if visited[each_point] != True: # if not visited
            visited[each_point] = True
            other_neighbor_points = region_query(coordinates, each_point, epsilon)
            if len(other_neighbor_points) >= min_points: # if not noise
                neighbor_points.extend(other_neighbor_points)
        if clusters[each_point] == 0: # if not a member of a cluster yet
            clusters[each_point] = C

# returns a list of all point(s) within epsilon distance of P, including P
def region_query(coordinates, P, epsilon):
    all_points_within_P = []
    point_one = coordinates[P] # coordinates of P

    for each_point in range(0, coordinates.shape[0]):
        point_two = coordinates[each_point] # coordinates of other points
        if np.linalg.norm(point_one - point_two) <= epsilon: # euclidean distance
            all_points_within_P.append(each_point)

    return all_points_within_P

# runs PCA on feature_matrix
def PCA(feature_matrix):
    # Adjusted Matrix
    adjMatrix = feature_matrix - feature_matrix.mean(axis = 0)
    # Covariance of adjusted matrix
    covMatrix = np.cov(adjMatrix.T)
    # Eigen value and Eigen vector
    eigenVal, eigenVec = np.linalg.eig(covMatrix)
    # Finding top n eigen value for principle component, In this case we are finding 2
    # -2 represents the top k values
    index = eigenVal.argsort()[-2:][::-1]
    eigenVal_top2 = eigenVal[index]
    eigenVec_top2 = eigenVec[:,index]
    # Need to build priniciple components from n dimensions to two dimensions
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

# returns a cluster matrix and ground truth matrix used for comparison
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

# returns jaccard similarity between ground truth and computer clusters
# def jaccard_similarity(ground_truth_clusters, clusters):
#     cluster_matrix, ground_truth_matrix = prepare_comparison(ground_truth_clusters, clusters)

#     one_and_one = 0.
#     one_or_one = 0. # one or one includes one and one

#     for row in range(0, cluster_matrix.shape[0]):
#         for col in range(0, cluster_matrix.shape[0]):
#             if cluster_matrix[row][col] == 1 or ground_truth_matrix[row][col] == 1:
#                 one_or_one += 1
#             if cluster_matrix[row][col] == 1 and ground_truth_matrix[row][col] == 1:
#                 one_and_one += 1

#     return one_and_one / one_or_one

# returns rand index between ground truth and computer clusters
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


# name_of_data_file  = input('Enter input data: ')
name_of_data_file = input("Enter name of file: ")
epsilon = float(input("Enter epsilon: "))
min_points = int(input("Enter min points: "))
data_file = np.loadtxt(name_of_data_file, dtype="float")

# extracts coordinates from point id and ground truth
coordinates = data_file[0:data_file.shape[0], 2:data_file.shape[1]]

# runs dbscan and returns cluster assignments, 0 cluster is noise
clusters = dbscan(coordinates, epsilon, min_points)

# comparison to ground truth
ground_truth_clusters = data_file[0:data_file.shape[0], 1]
#print ("Jaccard similarity: ", jaccard_similarity(ground_truth_clusters, clusters))
print ("Rand index: ", rand_index(ground_truth_clusters, clusters))

# prepares dbscan results appropriately to pass into Prerna's PCA plot_graph
feature_matrix = PCA(coordinates)
unique_cluster_number = set(clusters)
unique_cluster = {}
for n in range(0, len(unique_cluster_number)):
    unique_cluster[n] = n

# plots PCA graph
plot_graph(name_of_data_file, clusters, unique_cluster, unique_cluster_number, feature_matrix)