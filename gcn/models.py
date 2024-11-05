import numpy as np
import layers
import helpers
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class GCN:
    def __init__(self, n_features, hidden_dims, output_dim, init_method='glorot', use_bias=True, 
                 activation = helpers.ActivationFunctions.gelu, dropout_rate = 0.1, start_lr = 0.1, 
                 learning_rate = helpers.LearningRate.exp_lr, global_pool = None):
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.activation = activation
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.name=f'GCN_{n_features}x'
        # Initialize layers
        self.layers = []
        input_dim = n_features
        for hidden_dim in hidden_dims:
            self.layers.append(layers.GCNLayer(input_dim, hidden_dim, init_method, use_bias, start_lr, learning_rate))
            self.name+=str(hidden_dim)+'x'
            input_dim = hidden_dim
        
        self.name+=str(output_dim)
        if global_pool is not None:
            self.name+='_globalPool'
            self.layers.append(layers.GlobalPoolLayer(global_pool))
            self.layers.append(layers.DenseLayer(hidden_dim, output_dim, init_method, start_lr, learning_rate))
        else:
            self.layers.append(layers.GCNLayer(hidden_dim, output_dim, init_method, use_bias, start_lr, learning_rate))

    def forward(self, A, X, training = False):
        A_hat = helpers.MatrixHelper.normalize_adjacency_matrix(A)
        H = X
        for layer in self.layers[:-1]:
            H = self.activation(layer.forward(A_hat, H))
            if training:
                H = helpers.Regularization.dropout(H, dropout_rate=self.dropout_rate, training=training)
        H = self.layers[-1].forward(A_hat, H)
        H = helpers.ActivationFunctions.softmax(H)
        return H
    
    def get_name(self):
        return self.name
    
    def compute_gradients(self, predictions, targets, epoch):
        loss = helpers.LossFunctions.cross_entropy_loss(predictions, targets)
        
        dL_dH = helpers.LossFunctions.cross_entropy_loss_derivative(predictions, targets)
        
        for layer in reversed(self.layers):
            dL_dH = layer.backward(dL_dH, epoch)
        
        return loss
    
    def train_instance(self, A, X, Y, epoch, mask = None):
        predictions = self.forward(A, X, True)
        if mask is not None:
            loss = self.compute_gradients(predictions*mask, Y*mask, epoch)
            return loss
        loss = self.compute_gradients(predictions, Y, epoch)
        return loss
    
    def train(self, adjecancy, data, targets, epochs=100, early_stopping = None, mask = None):
        prev_loss = 0
        loss_graph_x = [x+1 for x in range(epochs)]
        loss_graph_y = []
        if len(data.shape) == 3 and len(adjecancy.shape) == 3:
            if data.shape[0] != adjecancy.shape[0]:
                print("Adjecancy and data have to have same number of matrices")
                return 
            
        for epoch in range(epochs):
            if len(data.shape) == 3 and len(adjecancy.shape) == 3: #If we have multiple matrices, multiple graphs - for graph classification
                total_loss = 0
                for i, X in enumerate(data):
                    total_loss += self.train_instance(adjecancy[i], X, targets[i], epoch, mask)
            else:
                total_loss = self.train_instance(adjecancy, data, targets, epoch, mask)

            loss_graph_y.append(total_loss)
            print(f"Epoch {epoch + 1}/{epochs}, Total Loss: {total_loss}")

            if early_stopping is not None and abs(total_loss-prev_loss) < early_stopping:
                return
            if np.isnan(total_loss):
                return
            prev_loss = total_loss
        
        plt.plot(loss_graph_x, loss_graph_y)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.show()

    def measure_accuracy(self, adjecancy, data, targets, mask = None):
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

        if mask is not None:
            select_mask = [i for i,x in enumerate(mask) if x[0] == 1]
            return helpers.Metrics.accuracy(targets[select_mask], predicted[select_mask])
        return helpers.Metrics.accuracy(targets, predicted)
    
    def confusion_matrix(self, adjecancy, data, targets, mask = None):
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
        if mask is not None:
            select_mask = [i for i,x in enumerate(mask) if x[0] == 1]
            targets = targets[select_mask]
            predicted = predicted[select_mask]

        y_pred = np.argmax(predicted, axis=1)
        
        if len(targets.shape) > 1:
            targets = np.argmax(targets, axis=1)

        cm = confusion_matrix(targets, y_pred)
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=True, yticklabels=True)
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title("Confusion Matrix")
        plt.show()