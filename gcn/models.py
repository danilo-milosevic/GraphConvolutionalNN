import numpy as np
import layers
import helpers

class GCN:
    def __init__(self, n_features, hidden_dims, output_dim, init_method='glorot', use_bias=True, activation = helpers.ActivationFunctions.gelu, start_lr = 0.1, learning_rate = helpers.LearningRate.exp_lr, global_pool = None):
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.activation = activation
        self.learning_rate = learning_rate
        
        # Initialize layers
        self.layers = []
        input_dim = n_features
        for hidden_dim in hidden_dims:
            self.layers.append(layers.GCNLayer(input_dim, hidden_dim, init_method, use_bias, start_lr, learning_rate))
            input_dim = hidden_dim
        if global_pool is not None:
            self.layers.append(layers.GlobalPoolLayer(global_pool))
            self.layers.append(layers.DenseLayer(input_dim, output_dim, init_method, start_lr, learning_rate))
        else:
            self.layers.append(layers.GCNLayer(input_dim, output_dim, init_method, use_bias, start_lr, learning_rate))

    def forward(self, A, X):
        A_hat = helpers.MatrixHelper.normalize_adjacency_matrix(A)
        H = X
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
    
    def train_instance(self, A, X, Y, epoch):
        predictions = self.forward(A, X)
        loss = self.compute_gradients(predictions, Y, epoch)
        return loss
        
    
    def train(self, adjecancy, data, targets, epochs=100, early_stopping = None):
        prev_loss = 0
        if len(data.shape) == 3 and len(adjecancy.shape) == 3:
            if data.shape[0] != adjecancy.shape[0]:
                print("Adjecancy and data have to have same number of matrices")
                return 
        for epoch in range(epochs):
            if len(data.shape) == 3 and len(adjecancy.shape) == 3: #If we have multiple matrices, multiple graphs - for graph classification
                total_loss = 0
                for i, X in enumerate(data):
                    total_loss += self.train_instance(adjecancy[i], X, targets[i], epoch)
            else:
                total_loss = self.train_instance(adjecancy, data, targets, epoch)
            print(f"Epoch {epoch + 1}/{epochs}, Total Loss: {total_loss}")
            if early_stopping is not None and abs(total_loss-prev_loss) < early_stopping:
                return
            prev_loss = total_loss

    def measure_accuracy(self, adjecancy, data, targets):
        predicted = []
        if len(data.shape) == 3 and len(adjecancy.shape) == 3: 
            #If we have multiple matrices, multiple graphs - for graph classification
            if data.shape[0] != adjecancy.shape[0]:
                print("Adjecancy and data have to have same number of matrices")
                return 
            for i, X in enumerate(data):
                predicted.append(self.forward(adjecancy[i], X))
        else:
            predicted = self.forward(adjecancy, data)
        return helpers.Metrics.accuracy(targets, predicted)