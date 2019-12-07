# V2A1_LinearRegression.py
# Programmgeruest zu Versuch 2, Aufgabe 1
import numpy as np
import matplotlib.pyplot as plt


# compute 1-dim. parable function; X must be Nx1 data matrix
def fun_true(X):
    # true parameters of parable y(x)=w0+w1*x+w2*x*x
    w2, w1, w0 = 3.0, -1.0, 2.0
    # return function values (same size as X)
    return w0+w1*X+w2*np.multiply(X, X)


# generate data matrix X and target values T
def generateDataSet(N, xmin, xmax, sd_noise):
    # get random x values uniformly in [xmin;xmax)
    X = xmin+np.random.rand(N, 1)*(xmax-xmin)
    T = fun_true(X)                            # target values without noise
    if(sd_noise > 0):
        T = T+np.random.normal(0, sd_noise, X.shape)  # add noise
    return X, T


# compute data error (least squares) between prediction Y and true target values T
def getDataError(Y, T):
    # squared differences between Y and T
    D = np.multiply(Y-T, Y-T)
    # return least-squares data error function E_D
    return 0.5*sum(sum(D))


# compute polynomial basis function vector phi(x) for data x
def phi_polynomial(x, deg=1):
    assert(np.shape(x) == (1,)), "currently only 1dim data supported"
    # returns feature vector phi(x)=[1 x x**2 x**3 ... x**deg]
    return np.array([x[0]**i for i in range(deg+1)]).T


# (I) generate data
# set seed of random generator (to be able to regenerate data)
np.random.seed(10)
N = 10                                          # number of data samples
xmin, xmax = -5.0, 5.0                            # x limits
# standard deviation of Guassian noise
sd_noise = 10
# generate training data
X, T = generateDataSet(N, xmin, xmax, sd_noise)
X_test, T_test = generateDataSet(
    N, xmin, xmax, sd_noise)             # generate test data
print("X=", X, "T=", T)

# (II) generate linear least squares model for regression
# no regression
lmbda = 0
# degree of polynomial basis functions
deg = 5
# shape of data matrix X
N, D = np.shape(X)
# shape of target value matrix T
N, K = np.shape(T)
# generate design matrix
PHI = np.array([phi_polynomial(X[i], deg).T for i in range(N)])
# shape of design matrix
N, M = np.shape(PHI)
print("PHI=", PHI)

PHITPHI_lambdaI_inv = np.linalg.inv(
    np.dot(np.transpose(PHI), PHI)+lmbda*np.eye(M))

W_LSR = np.dot(np.dot(PHITPHI_lambdaI_inv, np.transpose(PHI)), T)
print("W_LSR=", W_LSR)

# (III) make predictions for training and test data
ymin, ymax = -50.0, 150.0               # interval of y data
x_ = np.arange(xmin, xmax, 0.01)        # densely sampled x values
Y_train = np.zeros((N, 1))
Y_train = np.array([
    np.dot(W_LSR.T, np.array([phi_polynomial(X[i], deg)]).T)
    [0] for i in range(N)]
)       # least squares prediction

Y_test = np.zeros((N, 1))
Y_test = np.array([
    np.dot(W_LSR.T, np.array([phi_polynomial(X_test[i], deg)]).T)
    [0] for i in range(N)]
)       # least squares prediction
print("Y_test=", Y_test)
print("T_test=", T_test)
print("training data error = ", getDataError(Y_train, T))
print("test data error = ", getDataError(Y_test, T_test))
print("W_LSR=", W_LSR)
print("mean weight = ", np.mean(np.mean(np.abs(W_LSR))))

# (IV) plot data
ymin, ymax = -50.0, 150.0               # interval of y data
x_ = np.arange(xmin, xmax, 0.01)        # densely sampled x values
Y_LSR = np.array([np.dot(W_LSR.T, np.array([phi_polynomial([x], deg)]).T)[
                 0] for x in x_])       # least squares prediction
Y_true = fun_true(x_).flat

print("Y_LSR=", Y_LSR)


# v2a1c3
list_deg = [1, 2, 4, 6, 9, 11, 13]
mean_weights_wlsr = []
for d in list_deg:
    PHI = np.array([phi_polynomial(X[i], d).T for i in range(N)])
    N, M = np.shape(PHI)
    PHITPHI_lambdaI_inv = np.linalg.inv(
        np.dot(np.transpose(PHI), PHI)+lmbda*np.eye(M))
    WLSR = np.dot(np.dot(PHITPHI_lambdaI_inv, np.transpose(PHI)), T)
    m_weights = np.mean(np.abs(WLSR))

    mean_weights_wlsr = mean_weights_wlsr + [m_weights]

deg = 9
Ns = [10, 100, 1000, 10000]
for n in Ns:
    X, T = generateDataSet(n, xmin, xmax, sd_noise)
    X_test, T_test = generateDataSet(
        n, xmin, xmax, sd_noise)             # generate test data
    # print("X=", X, "T=", T)
    PHI = np.array([phi_polynomial(X[i], deg).T for i in range(n)])
    N, M = np.shape(PHI)
    PHITPHI_lambdaI_inv = np.linalg.inv(
        np.dot(np.transpose(PHI), PHI)+lmbda*np.eye(M))
    WLSR = np.dot(np.dot(PHITPHI_lambdaI_inv, np.transpose(PHI)), T)

    ymin, ymax = -50.0, 150.0               # interval of y data
    x_ = np.arange(xmin, xmax, 0.01)        # densely sampled x values
    Y_train = np.zeros((N, 1))
    Y_train = np.array([
        np.dot(WLSR.T, np.array([phi_polynomial(X[i], deg)]).T)
        [0] for i in range(n)]
    )       # least squares prediction
    Y_test = np.zeros((n, 1))
    Y_test = np.array([
        np.dot(WLSR.T, np.array([phi_polynomial(X_test[i], deg)]).T)
        [0] for i in range(n)]
    )       # least squares prediction
    print("N=", n)
    # print("Y_test=", Y_test)
    # print("T_test=", T_test)
    print("training data error = ", getDataError(Y_train, T)/n)
    print("test data error = ", getDataError(Y_test, T_test)/n)
    # print("W_LSR=", WLSR)
    print("mean weight = ", np.mean(np.abs(WLSR)))
    print("\n")

fig = plt.figure()
ax = fig.add_subplot(111)
# plot learning data points (green x)
ax.scatter(X.flat, T.flat, c='g', marker='x', s=100)
ax.scatter(X_test.flat, T_test.flat, c='g', marker='.',
           s=100)                       # plot test data points (green .)
ax.plot(x_, Y_LSR.flat, c='r')          # plot LSR regression curve (red)
ax.plot(x_, Y_true, c='g')              # plot true function curve (green)
ax.set_xlabel('x')                      # label on x-axis
ax.set_ylabel('y')                      # label on y-axis
ax.grid()                               # draw a grid
plt.ylim((ymin, ymax))                  # set y-limits
plt.show()                              # show plot on screen
