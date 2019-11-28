# Programmgeruest zu Versuch 1, Aufgabe 1
import numpy as np


# realizes nearest neighbor search of x in database
def getKNearestNeighbors(x, X, k=1):
    """
    compute the k nearest neighbors for a query vector x given a data matrix X
    :param x: the query vector x
    :param X: the N x D data matrix (in each row there is data vector) as a numpy array
    :param k: number of nearest-neighbors to be returned
    :return: return list of k line indixes referring to the k nearest neighbors of x in X
    """

    d = []  # init empty list
    for index in np.nditer(np.arange(len(X))):  # iterate through matrix X
        d.append(np.linalg.norm(X[index]-x))  # get euklidean distance

    return np.argsort(d)[:k]  # indexes of k smallest distances

# ***** MAIN PROGRAM ********


# (i) Generate dummy data
# data matrix X: list of data vectors (=database) of dimension D=3
X = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6]])
x = np.array([1.5, 3.6, 5.7])                          # a test data vector
print("Data matrix X=\n", X)
print("Test vector x=", x)

# (ii) Print all Euklidean distances to test vector x
# REPLACE! compute list of Euklidean distances
print("Euklidean distances to x: ", [
      np.linalg.norm(X[i]-x) for i in range(len(X))])

# (iii) Search for k nearest neighbor
k = 2
# get indexes of k nearest neighbors
idx_knn = getKNearestNeighbors(x, X, k)
print("idx_knn=", idx_knn)

# (iv) output results
print("The k Nearest Neighbors of x are the following vectors:")
for i in range(k):
    idx = idx_knn[i]
    print("The", i+1, "th nearest neighbor is: X[", idx, "]=",
          X[idx], " with distance ", np.linalg.norm(X[idx]-x))
