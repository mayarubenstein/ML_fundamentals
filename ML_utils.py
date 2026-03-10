from sklearn.datasets import fetch_california_housing as cali        
import numpy

data = cali()
X = data.data 
Y = data.target
print(X.shape)
print(Y.shape)

# now we must shhuffle data. To keep the features of each datum together,
# we use indices to access the feauture vectors

def normalize_and_split(X, Y, spl):
    if spl < 0 or spl > 1:
        raise Exception("split must be in (0, 1)")
    n = len(X)
    indices = numpy.arange(n) # use numpy arange to be consistent with numpy
    # and because range doesn't create an actual array that can be shuffled
    # think or numpy.arange() as array-range not the word "arranging"
    numpy.random.shuffle(indices)

    # now we do the train/test split
    split = int(spl * n) # make sure to cast to int
    train_idx = indices[:split]
    test_idx = indices[split:]
    X_train, X_test = X[train_idx], X[test_idx] # fancy indexing: get matrices
    # by inexing with arrays ("fancy indexing")
    Y_train, Y_test = Y[train_idx], Y[test_idx]

    # now normalize everything!
    means = numpy.mean(X_train, axis = 0) # axis = 0 means keep the first 
    # axis (keep rows take average over column)
    stdev = numpy.std(X_train, axis = 0)
    X_train_normal = (X_train - means)/stdev # can apply tof each entry row-wise
    # like this
    X_test_normal = (X_test - means)/stdev # use stdev and mean from train data
    # to normalize test data
    # add a column of ones to both the X train matrix and X test matrix
    # using the function hstack
    ones_train = numpy.ones((len(X_train_normal), 1))
    ones_test= numpy.ones((len(X_test_normal), 1))
    X_train_normal = numpy.hstack([ones_train, X_train_normal])
    X_test_normal = numpy.hstack([ones_test, X_test_normal])
    return (X_train_normal, X_test_normal, Y_train, Y_test)

def residuals(param, X_matrix, Y_vector):
    return Y_vector - numpy.dot(X_matrix, param)

def mse_loss(param, X_matrix, Y_vector):
    res = residuals(param, X_matrix, Y_vector)
    return numpy.sum(res **2 )/len(X_matrix)

def mse_loss_regularized(param, X_matrix, Y_vector, gimmel = 0):
    res = residuals(param, X_matrix, Y_vector)
    return numpy.sum(res **2 )/len(X_matrix) + gimmel * numpy.dot(param, param)

def mse_gradient_regularized(param, X_matrix, Y_matrix, gimmel = 0):
    reg = 2 * gimmel * param
    reg[0] = 0
    return (-2/len(X_matrix)) * numpy.dot(X_matrix.T, residuals(param, X_matrix, Y_matrix)) + reg

def GD(X_matrix, Y_vector, gradientFunction, iter, eta = 0.01, gimmel = 0):
    params = numpy.zeros(X_matrix.shape[1])
    for _ in range(iter):
        params = params - eta * gradientFunction(params, X_matrix, Y_vector, gimmel)
    return params