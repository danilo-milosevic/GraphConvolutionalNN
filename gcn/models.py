import numpy as np
import layers
import helpers

class GCN:
    def __init__(self, A, X, hidden_dims, output_dim, init_method='glorot', use_bias=True, activation = helpers.ActivationFunctions.gelu, start_lr = 0.1, learning_rate = helpers.LearningRate.exp_lr):
        self.A = A
        self.X = X
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.activation = activation
        self.learning_rate = learning_rate
        
        # Initialize layers
        self.layers = []
        input_dim = X.shape[1]
        for hidden_dim in hidden_dims:
            self.layers.append(layers.GCNLayer(input_dim, hidden_dim, init_method, use_bias, start_lr, learning_rate))
            input_dim = hidden_dim
        self.layers.append(layers.GCNLayer(input_dim, output_dim, init_method, use_bias, start_lr, learning_rate))

    def forward(self):
        A_hat = helpers.MatrixHelper.normalize_adjacency_matrix(self.A)
        H = self.X
        for layer in self.layers[:-1]:
            H = self.activation(layer.forward(A_hat, H))
        H = self.layers[-1].forward(A_hat, H)
        return H
    
    def compute_gradients(self, predictions, targets, epoch):
        loss = helpers.LossFunctions.cross_entropy_loss(predictions, targets)
        
        dL_dH = helpers.LossFunctions.cross_entropy_loss_derivative(predictions, targets)
        
        for layer in reversed(self.layers):
            dL_dH = layer.backward(dL_dH, epoch)
        
        return loss
    
    def train(self, targets, epochs=100, early_stopping = None):
        
        for epoch in range(epochs):
            predictions = self.forward()
            
            loss = self.compute_gradients(predictions, targets, epoch)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss}")
            if early_stopping is not None and loss < early_stopping:
                return