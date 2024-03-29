'''This is a neural network implementation from scratch, designed to showcase learning
I take heavy inspiration from: [https://www.youtube.com/watch?v=w8yWXqWQYmU&t=39s&ab_channel=SamsonZhang].
For the sake of this project, I implement a basic, dense neural network with a hidden layer.'''

#the dataset used here is the MNIST digit dataset, parameter count of 7840 + 100 + 10 + 10 = 7960
#note that @ and .dot() dont behave the same, since @ maps to mat_mul, not .dot()
'''the architecture of this nn is as follows: 
784 node input layer (28 x 28 pixel images, 1 node per pixel)
10 node hidden layer
10 node output layer (corresponds to the 0-9 digits)
'''

#data prep block
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
data = pd.read_csv("/Users/viyer/Desktop/mlhub/nnfromscratch/digit-recognizer/train.csv")
print(data.head())
#data = np.array(data) #convert to array form to access easier
m, n = data.shape
X = data.iloc[:, 1:].values / 255
Y = data.iloc[:, 0].values
x_train, x_validation, y_train, y_validation = train_test_split(X, Y, test_size = 0.33, random_state = 42)
print(x_train.shape)
x_train = x_train.T
x_validation = x_validation.T
y_train = y_train.T
y_validation = y_validation.T


# -----------------------------------------------

def initialize_parameters():
    w1 = np.random.rand(10, 784) - 0.5 #rescale to -0.5, 0.5
    b1 = np.random.rand(10, 1) - 0.5
    w2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    return w1, b1, w2, b2

# -----------------------------------------------

def ReLU(z):
    return np.maximum(z, 0)

# -----------------------------------------------

def softmax(z):
    #prevents overflow errors
    z_exp = np.exp(z - np.max(z))
    return z_exp / z_exp.sum(axis = 0, keepdims = True)

#-------------------------------------------------

def onehotencoding(y):
    y_ohe = np.zeros((y.size, y.max() + 1))
    y_ohe[np.arange(y.size), y] = 1
    return y_ohe.T #return transposed version for each column to be an example, not each row

#-------------------------------------------------

def derivative_of_relu(z):
    return z > 0
    #technically 0 is a subdifferential at the corner, but it's used for simplicity
 
#-------------------------------------------------

def get_predictions(a2):
    return np.argmax(a2, 0)

#-------------------------------------------------

def get_accuracy(predictions, y):
    return np.sum(predictions == y) / y.size

#-------------------------------------------------

def forward_propagation(w1, b1, w2, b2, x):
    z1 = w1.dot(x) + b1
    a1 = ReLU(z1)
    z2 = w2.dot(a1) + b2
    a2 = softmax(z2)
    return z1, a1, z2, a2

#-------------------------------------------------

def backward_propagation(z1, a1, z2, a2, w2, x, y):
    y_ohe = onehotencoding(y)
    dz2 = a2 - y_ohe
    dw2 = 1 / y.size * dz2.dot(a1.T)
    db2 = 1 / y.size * np.sum(dz2)
    dz1 = w2.T.dot(dz2) * derivative_of_relu(z1)
    dw1 = 1 / y.size * dz1.dot(x.T)
    db1 = 1 / y.size * np.sum(dz1)
    return dw1, db1, dw2, db2

#-------------------------------------------------

def update_parameters_post_back_prop(w1, b1, w2, b2, dw1, db1, dw2, db2, alpha):
    w1 = w1 - alpha * dw1
    b1 = b1 - alpha * db1
    w2 = w2 - alpha * dw2
    b2 = b2 - alpha * db2
    return w1, b1, w2, b2

#-------------------------------------------------

def gradient_descent(x, y, iterations, alpha):
    #init params then run fwd prop, back prop, update params loop (the number given by iterations) times
    w1, b1, w2, b2 = initialize_parameters()
    for i in range(iterations+1):
        z1, a1, z2, a2 = forward_propagation(w1, b1, w2, b2, x)
        dw1, db1, dw2, db2 = backward_propagation(z1, a1, z2, a2, w2, x, y)
        w1, b1, w2, b2 = update_parameters_post_back_prop(w1, b1, w2, b2, dw1, db1, dw2, db2, alpha)
        if (i % 100  == 0):
            #every 100th iteration print loss
            print("Iteration: ", i)
            print("Accuracy: ", get_accuracy(get_predictions(a2), y)) 
    return w1, b1, w2, b2


#-------------------------------------------------

#Assignment Block

w1, b1, w2, b2 = gradient_descent(x_train, y_train, iterations = 1000, alpha = 0.1) 



