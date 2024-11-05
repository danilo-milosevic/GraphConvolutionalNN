import numpy as np
class ActivationFunctions:
    @staticmethod
    def relu(x):
        return np.maximum(0, x)
    @staticmethod
    def gelu(x):
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    @staticmethod
    def swish(x):
        return x * ActivationFunctions.sigmoid(x)
    @staticmethod
    def leaky_relu(x, negative_slope=0.2):
        return np.maximum(negative_slope * x, x)
    @staticmethod   
    def softmax(x, axis=-1):
        exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))  # For numerical stability
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

class LossFunctions:
    @staticmethod
    def cross_entropy_loss(predictions, targets):
        # Avoid log(0) by clipping predictions
        epsilon = 1e-6
        predictions = np.clip(predictions, epsilon, 1. - epsilon)
        n = targets.shape[0]
        return -np.sum(targets * np.log(predictions)) / n
    @staticmethod
    def cross_entropy_loss_derivative(predictions, targets):
        # Derivative of cross-entropy loss with respect to predictions
        epsilon = 1e-6
        predictions = np.clip(predictions, epsilon, 1. - epsilon)
        n = targets.shape[0]
        return (predictions-targets)/n
    
class LearningRate:
    @staticmethod
    def static_lr(epoch, start_rate):
        return start_rate
    @staticmethod
    def exp_lr(epoch, start_rate):
        k = 0.1
        return start_rate * np.exp(-k*epoch)
    
class MatrixHelper:
    @staticmethod
    def normalize_adjacency_matrix(A):
        I = np.eye(A.shape[0])
        A_hat = A + I
        D_hat = np.diag(np.sum(A_hat, axis=1))
        D_hat_inv_sqrt = np.linalg.inv(np.sqrt(D_hat))
        return D_hat_inv_sqrt @ A_hat @ D_hat_inv_sqrt
    
class Metrics:
    @staticmethod
    def accuracy(y_true, y_pred_probs):
        y_pred = np.argmax(y_pred_probs, axis=1)
        
        if len(y_true.shape) > 1:
            y_true = np.argmax(y_true, axis=1)

        correct_predictions = np.sum(y_true == y_pred)
        total_predictions = y_true.shape[0]
        
        accuracy = correct_predictions / total_predictions
        return accuracy
    
class Regularization:
    @staticmethod
    def dropout(X, dropout_rate=0.5, training=True):
        if not training or dropout_rate == 0.0:
            return X
        mask = np.random.binomial(1, 1 - dropout_rate, size=X.shape)
        return X * mask / (1 - dropout_rate)