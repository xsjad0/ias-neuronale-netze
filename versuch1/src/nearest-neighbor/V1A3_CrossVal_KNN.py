#!/usr/bin/env python
# Programmgeruest zu Versuch 1, Aufgabe 3
import numpy as np
import scipy.spatial
from random import randint
import matplotlib
import matplotlib.pyplot as plt
from time import clock
from V1A2_Classifier import *

# (i) create some synthetic data (2-dimensional Gaussian)
C = 2                               # two classes
N1, N2 = 500, 500                     # N1 and N2 data vectors for the two classes
mu1, mu2 = [1, 1], [3, 1]           # expectations for the two classes
sigma1 = [[1, 0.5],
          [0.5, 1]]                # covariance matrix for class 1
sigma2 = [[1, 0.5],
          [0.5, 1]]                # covariance matrix for class 2
# Gaussian data vectors for class 1
X1 = np.random.multivariate_normal(mu1, sigma1, (N1))
# Gaussian data vectors for class 2
X2 = np.random.multivariate_normal(mu2, sigma2, (N2))
T1, T2 = N1*[0], N2*[1]            # corresponding class labels
X = np.concatenate((X1, X2))       # entire data set
T = np.concatenate((T1, T2))       # entire label set
N, D = X.shape[0], X.shape[1]      # size of data set
print("Data size: N=", N, ", D=", D)


def task():
    # (ii) create and test classifiers
    # k=number of nearest neighbors; S=number of data subsets for cross validation
    for k in [1, 5, 11]:
        for S in [1, 2, 5]:

            # Some additional data vectors to be tested
            X_test = np.array([[2, 1], [5, 1], [-1, 1]])

            # (ii.a) test of naive KNN classifier
            print("\nNaive KNN Classifier:", "\n------------------------")
            # create classifier object of class KNNClassifier
            knnc = KNNClassifier(C, k)
            t1 = clock()                        # start time
            # do S-fold cross validation for data X,T
            pE_naive, pCE_naive = knnc.crossvalidate(S, X, T)
            t2 = clock()                        # end time
            # wall time required by the naive KNN algorithmus (in seconds)
            t_naive = t2-t1
            print("S=", S, " fold Cross-Validation of naive ", k, "-NN-Classifier requires ",
                  t_naive, " seconds. Confusion error probability matrix is \n", pCE_naive)
            print("Probability of a classification error is pE = ", pE_naive)
            # train classifier with whole data set
            knnc.fit(X, T)
            for x_test in X_test:             # Test some additional data vectors x_test from X_test
                t_test, p_class, idxNN = knnc.predict(x_test, k)
                print("New data vector x_test=", x_test, " is most likely from class ",
                      t_test, "; class probabilities are p_class = ", p_class)

            # (ii.b) test of KD-tree KNN classifier
            print("\nFast KNN Classifier based on KD-Trees:",
                  "\n---------------------------------------")
            # create classifier object of class KNNClassifier
            fknnc = FastKNNClassifier(C, k)
            t1 = clock()                        # start time
            # do S-fold cross validation for data X,T
            pCE_kdtree, pE_kdtree = fknnc.crossvalidate(S, X, T)
            t2 = clock()                        # end time
            # wall time required by the naive KNN algorithmus (in seconds)
            t_kdtree = t2-t1
            print("S=", S, " fold Cross-Validation of fast ", k, "-NN-Classifier requires ",
                  t_kdtree, " seconds. Confusion error probability matrix is \n", pE_kdtree)
            print("Probability of a classification error is pE = ", pCE_kdtree)

            knnc.fit(X, T)
            for x_test in X_test:             # Test some additional data vectors x_test from X_test
                t_test, p_class, idxNN = knnc.predict(x_test, k)
                print("New data vector x_test=", x_test, " is most likely from class ",
                      t_test, "; class probabilities are p_class = ", p_class)

# (iii) plot data


def plot():
    xlabel = 'feature x1'
    ylabel = 'feature x2'
    f = plt.figure()

    a = f.add_subplot(111)
    # plot data vectors of class 1
    a.plot(X1.T[0], X1.T[1], 'rx')
    # plot data vectors of class 2
    a.plot(X2.T[0], X2.T[1], 'g+')
    a.set_xlabel(xlabel)
    a.set_ylabel(ylabel)
    # a.set_title('Naive: '+str(t_naive)+'sec/ KD-Tree: '+str(t_kdtree) +
    #             'sec; Classification Error='+str(pE_naive)+'/'+str(pE_kdtree))
    a.set_title('Gaussian data vectors for class 1 and 2')

    # plt.show()
    plt.savefig('plot.pdf')


if __name__ == "__main__":
    # plot()
    task()
