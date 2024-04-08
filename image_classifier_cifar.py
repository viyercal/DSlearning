'''The objective of this program is to classify images using conv nets
dataset citation: Krizhevsky, A., Nair, V., & Hinton, G. (2009). 
Learning multiple layers of features from tiny images. Unpublished manuscript, University of Toronto.
'''


import torch
import numpy as np
import pandas as pd
import sklearn
import torchvision
from torchvision import transforms, datasets
import pickle
import torch.nn as nn
from torch.nn import Module
from torch.nn import Conv2d
from torch.nn import Linear
from torch.nn import MaxPool2d
from torch.nn import ReLU
from torch.nn import LogSoftmax
from torch.nn import Sigmoid
from torch import flatten
from scipy import signal

#----------------------------------------------------------------------------------
#Standardization Block for Images

data_transforms = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.RandomHorizontalFlip(),  
    transforms.RandomAffine(degrees=0, translate=None, scale=(0.8, 1.2), shear=0.2),  
    transforms.ToTensor(), 
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

#----------------------------------------------------------------------------------

#Loading Image Files Block

def load_cifar_batch(filename):
    with open(filename, 'rb') as f:
        batch_dict = pickle.load(f, encoding='bytes')
        data = batch_dict[b'data']
        labels = batch_dict[b'labels']
        # Reshape & transpose data to the format (#images, height, width, channels)
        data = data.reshape(len(data), 3, 32, 32).transpose(0, 2, 3, 1)
        labels = np.array(labels)
        return data, labels

data_dir = '/Users/viyer/Desktop/mlhub/cifar-10-batches-py'

batch_files = [
    f'{data_dir}/data_batch_1',
    f'{data_dir}/data_batch_2',
    f'{data_dir}/data_batch_3',
    f'{data_dir}/data_batch_4',
    f'{data_dir}/data_batch_5'
]
test_batch_file = f'{data_dir}/test_batch'

train_data, train_labels = [], []
for batch_file in batch_files:
    data, labels = load_cifar_batch(batch_file)
    train_data.append(data)
    train_labels.append(labels)

train_data = np.concatenate(train_data)
train_labels = np.concatenate(train_labels)
print(train_data, train_labels)

test_data, test_labels = load_cifar_batch(test_batch_file)


#----------------------------------------------------------------------------------

#Setup for the Conv Net Block: - This is pretty much directly from: 
#https://www.youtube.com/watch?v=Lakz2MoHy6o&ab_channel=TheIndependentCode
#implemented to learn how convolutional layers work under-the-hood
class Convolutional_Layer():
    def __init__(self, input_shape, kernel_size, depth):
        #input_shape is a tuple containing the depth, the height, and the width of the input
        #kernel_size is a number representing the size of each matrix inside each kernel
        #depth is the number of kernels we want, and thus the depth of the output
        input_depth, input_height, input_width = input_shape
        self.depth = depth
        self.input_shape = input_shape
        self.input_depth = input_depth
        self.output_shape = (depth, input_height - kernel_size + 1, input_width - kernel_size + 1)
        #number of kernels, height and width of the output matrix (size of input - size of kernel + 1)
        self.kernels_shape = (depth, input_depth, kernel_size, kernel_size)
        #4d kernel, where each kernel is a 3 dimensional block. of the four dimensions:
        #first dimension is the number of kernels, 
        #second dimension is the depth of each kernel (which is the depth of the input)
        #last two dimensions represent the size of the matrices contained within each kernel
        self.kernels = np.random.randn(*self.kernels_shape)
        self.biases = np.random.randn(*self.output_shape)
        #initialize these randomly
        #biases have the shape of the outputs
    def forward_propagation(self, input):
        self.input = input
        self.output = np.copy(self.biases)
        #since each output is equal to the bias plus something else, and we can initialize here to just bias
        for i in range(self.depth):
            #iterate over output depth
            for x in range(self.input_depth):
                #iterate over input_depth and add values to the output
                self.output[i] += signal.correlate2d(self.input[x], self.kernels[i, x], "valid")
                '''use the valid cross-correlation method that creates lower dimensional outputs from input
                as opposed to the full cross-correlation method that expands output size beyond input_shape. 
                Note cross-correlation is different than convolution, convolution simply rotates the kernel
                in cross-correlation by 180 degrees'''
        return self.output
    def back_propagation(self, output_gradient, learning_rate):
        kernel_gradient = np.zeros(self.kernels_shape)
        input_gradient = np.zeros(self.input_shape)
        for i in range(self.depth):
            for x in range(self.input_depth):
                kernel_gradient[i, x] = signal.correlate2d(self.input[x], output_gradient[x], "valid")
                input_gradient[x] += signal.convolve2d(output_gradient[i], self.kernels[i, x], "full")
                #bias gradient doesn't need to be calculated in the for loop since its equal to the output_gradient
        self.kernels -= learning_rate * kernel_gradient
        self.biases -= learning_rate * output_gradient
        return input_gradient
    
network = nn.Sequential(
    Convolutional_Layer(input_shape=(3, 150, 150), kernel_size=3, depth=32),
    #padding different than other layer implementations in Pytorch, so keep in mind for design of other layers
    ReLU(),
    MaxPool2d(kernel_size=2, stride= 2),
    Convolutional_Layer(input_shape=(32, 74, 74), kernel_size=3, depth=64),
    Convolutional_Layer(input_shape=(64, 36, 36), kernel_size=3, depth=128),
    Linear(128 * 17 * 17, 512),
    Linear(512, 10)
    )

    
    

#----------------------------------------------------------------------------------

#Training the Conv Net on CIFAR

#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------
