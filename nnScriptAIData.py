import numpy as np
import pickle
import random
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt


def initializeWeights(n_in, n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer

    # Output:
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""

    epsilon = sqrt(6) / sqrt(n_in + n_out + 1)
    W = (np.random.rand(n_out, n_in + 1) * 2 * epsilon) - epsilon
    return W


def sigmoid(z):
    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""

    return 1 / (1 + np.exp(-z))


def preprocess():
    """ Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the
       training set
     test_data: matrix of training set. Each row of test_data contains
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set

     Things to do for preprocessing step:
     - remove features that have the same value for all data points
           with corresponding labels
     - convert original data set from integer to double by using double()
           function
     - normalize the data to [0, 1]
     - divide the original data set to training, validation and testing set"""

    # Preparing the data set
    with open('AI_quick_draw.pickle', 'rb') as open_ai_quick:
        train_data = pickle.load(open_ai_quick)
        train_label = pickle.load(open_ai_quick)
        test_data = pickle.load(open_ai_quick)
        test_label = pickle.load(open_ai_quick)

    # remove features that have same value for all points in the training data
    # convert data to double
    # normalize data to [0,1]


        # remove features that have same value for all points in the training data
        to_remove = np.all(train_data == train_data[0, :], axis=0)
        train_data = train_data[:, ~to_remove]
        test_data = test_data[:, ~to_remove]

        # convert data to double
        train_data = train_data.astype(float)
        test_data = test_data.astype(float)
        train_label = train_label.astype(int)

        # normalize data to [0,1]
        train_data = train_data / 255
        test_data = test_data / 255


        # Split train_data and train_label into train_data, validation_data and train_label, validation_label
        # replace the next two lines

        # creating a list of 10,000 indexes to remove from training data, and insert into validation data and validation label
        to_validation = random.sample(range(0, 100000), 15000)
        validation_data = train_data[to_validation]
        validation_label = train_label[to_validation]

        # removing validation images from training data
        train_data = np.delete(train_data, to_validation, axis=0)
        train_label = np.delete(train_label, to_validation, axis=0)

        print(len(validation_label))
        print(len(train_data))






    print("preprocess done!")

    return train_data, train_label, validation_data, validation_label, test_data, test_label


def nnObjFunction(params, *args):
    """% nnObjFunction computes the value of objective function (negative log
    %   likelihood error function with regularization) given the parameters
    %   of Neural Networks, thetraining data, their corresponding training
    %   labels and lambda - regularization hyper-parameter.

    % Input:
    % params: vector of weights of 2 matrices w1 (weights of connections from
    %     input layer to hidden layer) and w2 (weights of connections from
    %     hidden layer to output layer) where all of the weights are contained
    %     in a single vector.
    % n_input: number of node in input layer (not include the bias node)
    % n_hidden: number of node in hidden layer (not include the bias node)
    % n_class: number of node in output layer (number of classes in
    %     classification problem
    % training_data: matrix of training data. Each row of this matrix
    %     represents the feature vector of a particular image
    % training_label: the vector of truth label of training images. Each entry
    %     in the vector represents the truth label of its corresponding image.
    % lambda: regularization hyper-parameter. This value is used for fixing the
    %     overfitting problem.

    % Output:
    % obj_val: a scalar value representing value of error function
    % obj_grad: a SINGLE vector of gradient value of error function
    % NOTE: how to compute obj_grad
    % Use backpropagation algorithm to compute the gradient of error function
    % for each weights in weight matrices.

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % reshape 'params' vector into 2 matrices of weight w1 and w2
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input
    %     layer to unit i in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in hidden
    %     layer to unit i in output layer."""

    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args

    w1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    obj_val = 0

    # Your code here
    #
    #
    #
    #
    #



    # Forward propogation
    # adding biases to input data with the size of training_data columns
    training_data = np.column_stack((training_data, np.ones(training_data.shape[0])))

    # Multiplying and making the function non linear
    zj_hidden1 = sigmoid(np.dot(training_data, w1.T))

    # adding bias to hidden layer1 with the size of zj_hidden1 columns
    zj_hidden1 = np.column_stack((zj_hidden1, np.ones(zj_hidden1.shape[0])))
    # Multiplying and making the function non linear
    outputs = sigmoid(np.dot(zj_hidden1, w2.T))

    # 1 to k encoding
    new_training_label = np.zeros((len(training_data), 10))


    for i in range(len(new_training_label)):
        new_training_label[i][train_label[i][0] - 1] = 1

    # Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2
    # you would use code similar to the one below to create a flat array
    # obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)
    # Back propogation
    delta_l = outputs - new_training_label

    grad_w2 = np.dot(delta_l.T, zj_hidden1)
    grad_w1 = np.dot(((1 - zj_hidden1) * zj_hidden1 * (np.dot(delta_l, w2))).T, training_data)

    # Remove zero row
    grad_w1 = np.delete(grad_w1, n_hidden, 0)

    num_samples = training_data.shape[0]

    # obj_grad
    obj_grad = np.array([])
    obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()), 0)
    obj_grad = obj_grad / num_samples

    # obj_val
    obj_val_part1 = np.sum(-1 * (new_training_label * np.log(outputs) + (1 - new_training_label) * np.log(1 - outputs)))
    obj_val_part1 = obj_val_part1 / num_samples
    obj_val_part2 = (lambdaval / (2 * num_samples)) * (np.sum(np.square(w1)) + np.sum(np.square(w2)))
    obj_val = obj_val_part1 + obj_val_part2

    return (obj_val, obj_grad)


def nnPredict(w1, w2, data):
    """% nnPredict predicts the label of data given the parameter w1, w2 of Neural
    % Network.

    % Input:
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit i in input
    %     layer to unit j in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit i in input
    %     layer to unit j in hidden layer.
    % data: matrix of data. Each row of this matrix represents the feature
    %       vector of a particular image

    % Output:
    % label: a column vector of predicted labels"""

    # Add bias
    data = np.column_stack((data, np.ones(data.shape[0])))
    zj_array_n = sigmoid(np.dot(data, w1.T))
    # Add bias
    zj_array_n = np.column_stack((zj_array_n, np.ones(zj_array_n.shape[0])))
    # Feed to output
    ol_array_n = sigmoid(np.dot(zj_array_n, w2.T))

    # Return indices of max as labels
    labels = np.argmax(ol_array_n, axis=1) + 1


    labels = labels.reshape(-1, 1)

    return labels


"""**************Neural Network Script Starts here********************************"""

train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()

#  Train Neural Network

# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]

# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 30

# set the number of nodes in output unit
n_class = 10

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden)
initial_w2 = initializeWeights(n_hidden, n_class)

# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()), 0)

# set the regularization hyper-parameter
lambdaval = 0.1

args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

# Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example

opts = {'maxiter': 50}  # Preferred value.

nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)

# In Case you want to use fmin_cg, you may have to split the nnObjectFunction to two functions nnObjFunctionVal
# and nnObjGradient. Check documentation for this function before you proceed.
# nn_params, cost = fmin_cg(nnObjFunctionVal, initialWeights, nnObjGradient,args = args, maxiter = 50)


# Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

# Test the computed parameters

predicted_label = nnPredict(w1, w2, train_data)

# find the accuracy on Training Dataset

for i in range(len(train_label)):
    print("predicted: " + str(predicted_label[i][0]) + " actual: " + str(train_label[i][0]))

print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')

predicted_label = nnPredict(w1, w2, validation_data)

# find the accuracy on Validation Dataset

print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label).astype(float))) + '%')

predicted_label = nnPredict(w1, w2, test_data)

# find the accuracy on Validation Dataset

print('\n Test set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')
