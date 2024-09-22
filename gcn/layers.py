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
        if init_method in init_method_dict.keys():
            init_method = 'glorot'
            print('Unknown init method, using glorot...')
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