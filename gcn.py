from typing import List
import numpy as np
from matplotlib import pyplot as plt
import os
import pandas as pd


class GraphHelper:
    def __init__(self):
        pass

    def generate_random_graph_for_testing(self, num_nodes:int = 10, n_dim:int = 5, n_classes:int = 5, edge_proba:float = 0.9):
        if edge_proba <= 0 or edge_proba > 1:
            raise Exception("Edge probability must be >0 and <=1")
        
        X = np.random.rand(num_nodes, n_dim)
        y_node = np.random.randint(0, n_classes, size =(num_nodes,))
        y_edge = []
        A = np.zeros((num_nodes, num_nodes))
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i==j:
                    continue
                if A[i][j]==1:
                    continue
                throw = np.random.random()
                if throw>=edge_proba:
                    A[i][j] = 1
                    A[j][i] = 1
                    y_edge.append((i, j, np.random.randint(0, n_classes)))
        return X, y_node, y_edge, A
        

    def get_normalized_graph_laplacian(self, A: np.ndarray) -> np.ndarray:
        diag_mat = np.diag(np.sum(A,axis=1))
        try:
            d = np.linalg.inv(np.sqrt(diag_mat))
        except:
            d = np.linalg.pinv(np.sqrt(diag_mat))
        return np.identity(A.shape[0]) - np.dot(d, np.dot(A, d))
    
    def relu(self, x: np.ndarray) -> np.ndarray:
        map = x > 0
        return x * map
    
    def leaky_relu(self, x: np.ndarray, alpha: float)->np.ndarray:
        map = ((x > 0) - 0.5) * 2 * alpha
        return x * map
    
    def loss_function(self, y_true: np.ndarray, y_pred: np.ndarray)->float:
            return -np.mean(np.sum(y_true * np.log(y_pred + 1e-8), axis=1))
    
    def softmax(self, x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


class GCNLayer:
    def __init__(self, in_feat:int, out_feat: int):
        self.weight = self.glorot_init(in_feat, out_feat)
        self.bias = np.zeros((1, out_feat))
        self.A = None
        self.X = None
        self.g = GraphHelper()

    def glorot_init(self, n_in:int, n_out:int)->np.ndarray:
        std = 2/(n_in+n_out)
        return np.random.randn(n_in, n_out)* (std**1/2)
   
    def forward(self, X: np.ndarray, A: np.ndarray):
        self.A = A
        self.X = X
        L = self.g.get_normalized_graph_laplacian(A)
        return np.dot(np.dot(L, X), self.weight) + self.bias
    
    def backward(self, error: np.ndarray, lr):
        L = self.g.get_normalized_graph_laplacian(self.A)
        feat = np.dot(self.X.T, L)
        grad_weight = np.dot(feat, error)
        grad_bias = np.sum(error, axis=0, keepdims=True)
        grad_input = np.dot(L.T, np.dot(error, self.weight.T))

        self.weight -= lr * grad_weight
        self.bias -= lr * grad_bias
        return grad_input

class GraphConvNet:
    def __init__(self, in_feat:int, hid_feat:int, out_feat:int, layers:int=1):
        self.init_layer:GCNLayer = GCNLayer(in_feat, hid_feat)
        self.hidden_layer: List[GCNLayer] = []
        for _ in range(layers):
            self.hidden_layer.append(GCNLayer(hid_feat, hid_feat))
        self.out_layer:GCNLayer = GCNLayer(hid_feat, out_feat)
        self.layers = layers
        self.g = GraphHelper()

    def forward(self, X: np.ndarray, A: np.ndarray):
        H = self.g.leaky_relu(self.init_layer.forward(X, A), 0.1)
        for i in range(self.layers):
            H = self.g.leaky_relu(self.hidden_layer[i].forward(H, A), 0.1)
        return self.g.softmax(self.out_layer.forward(H, A))

    def backward(self, y_true, y_predicted, epoch, alpha=0.01):
        def exp_alpha(epoch):
            return np.exp(-(epoch**1/2))
        
        # alpha = exp_alpha(epoch)
        error = (y_true - y_predicted) / y_true.shape[0]
        gradient_out = self.out_layer.backward(error, alpha)
        for i in range(self.layers-1, -1, -1):
            gradient_hid = self.hidden_layer[i].backward(gradient_out, alpha)
        gradient_init = self.init_layer.backward(gradient_hid, alpha)

        return gradient_init, gradient_hid, gradient_out

gh = GraphHelper()
X,y_n, y_e,A = gh.generate_random_graph_for_testing()

num_labels = 5

y = np.eye(num_labels)[y_n]

input_dim = X.shape[1]
hidden_dim = 16
output_dim = num_labels
epochs = 100
lr = 0.01

gcn = GraphConvNet(input_dim, hidden_dim, output_dim, layers=7)
g = GraphHelper()

loss_list = []
for epoch in range(epochs):
    y_hat = gcn.forward(X, A)

    loss = g.loss_function(y, y_hat)
    loss_list.append(loss)
    print(f"the epoch {epoch+1}/{epochs} : \n The current Loss => {loss}")
    gcn.backward(y, y_hat, alpha=lr, epoch=epoch)
print("train finished")

plt.plot(range(epochs), loss_list)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss Curve")
plt.grid(True)  # Add grid lines for better readability
plt.show()