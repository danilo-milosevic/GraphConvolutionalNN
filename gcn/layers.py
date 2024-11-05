import numpy as np
import helpers

class GCNLayer:
    
    def glorot_initialization(self, input_dim, output_dim):
        limit = np.sqrt(6 / (input_dim + output_dim))
        return np.random.uniform(-limit, limit, (input_dim, output_dim))
    
    def he_initialization(self, input_dim, output_dim):
        stddev = np.sqrt(2 / (input_dim))
        return np.random.randn(input_dim, output_dim) * stddev
    
    def __init__(self, input_dim, output_dim, init_method = 'glorot', use_bias = True, start_lr = 0.1, learning_rate = helpers.LearningRate.exp_lr):
        init_method_dict = {
            'glorot':self.glorot_initialization,
            'he':self.he_initialization,
        }
        if init_method not in init_method_dict.keys():
            init_method = 'glorot'
        self.W = init_method_dict[init_method](input_dim, output_dim)
        self.use_bias = use_bias
        if self.use_bias:
            self.bias = np.zeros(output_dim)
        self.learning_rate = learning_rate
        self.start_lr = start_lr

    def forward(self, A_hat, H):
        self.A_hat = A_hat
        self.h_prev = H

        H_prime = H@self.W
        H_new = A_hat @ H_prime

        if self.use_bias:
            H_new += self.bias

        return H_new
    
    def backward(self, dH, epoch):
        dH_prime = dH @ self.W.T
        self.dW = self.h_prev.T @ (self.A_hat @ dH)

        if self.use_bias:
            self.db = np.sum(dH, axis=0)
        
        self.W -= self.learning_rate(epoch, self.start_lr) * self.dW
        if self.use_bias:
            self.bias -= self.learning_rate(epoch, self.start_lr) * self.db

        return dH_prime

class GlobalPoolLayer:
    def sum(self, X):
        return np.sum(X, axis = 0, keepdims=True)
    def max(self, X):
        return np.max(X, axis = 0, keepdims=True)
    def avg(self, X):
        return np.mean(X, axis = 0, keepdims=True)
    
    def __init__(self, method='sum'):
        if method not in ['sum','avg','max']:
            print("Unknown global pool method")
            return
        method_map = {
            'sum':self.sum,
            'max':self.max,
            'avg':self.avg,
        }
        self.method = method_map[method]
    
    def forward(self, A_hat, H):
        self.input = H
        return self.method(H)
    
    def backward(self, dH, epoch):
        return np.zeros_like(self.input)
        
class DenseLayer:
    def glorot_initialization(self, input_dim, output_dim):
            limit = np.sqrt(6 / (input_dim + output_dim))
            return np.random.uniform(-limit, limit, (input_dim, output_dim))
        
    def he_initialization(self, input_dim, output_dim):
        stddev = np.sqrt(2 / (input_dim))
        return np.random.randn(input_dim, output_dim) * stddev
    
    def __init__(self, input_size, output_size, init_method = 'glorot', start_lr = 0.1, learning_rate = helpers.LearningRate.exp_lr):
        # Initialize weights and biases
        
        init_method_dict = {
            'glorot':self.glorot_initialization,
            'he':self.he_initialization,
        }

        if init_method not in init_method_dict.keys():
            init_method = 'glorot'
            print('Unknown init method, using glorot...')

        self.W = init_method_dict[init_method](input_size, output_size)
        self.biases = np.zeros((1, output_size))
        self.learning_rate = learning_rate
        self.start_lr = start_lr
        self.input = None
        self.output = None

    def forward(self, A_hat, input_data):
        self.input = input_data
        self.output = np.dot(self.input, self.W) + self.biases
        return helpers.ActivationFunctions.softmax(self.output)

    def backward(self, dH, epoch):
        # Compute the gradient of the weights and biases
        input_gradient = np.dot(dH, self.W.T)
        weights_gradient = np.dot(self.input.T, dH)
        biases_gradient = np.sum(dH, axis=0, keepdims=True)

        # Update weights and biases
        self.W -= self.learning_rate(epoch, self.start_lr) * weights_gradient
        self.biases -= self.learning_rate(epoch, self.start_lr) * biases_gradient

        return input_gradient