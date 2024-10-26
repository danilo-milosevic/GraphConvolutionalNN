import numpy as np
import models
A = np.array([
    [[0, 1, 0, 0],
     [1, 0, 1, 1],
     [0, 1, 0, 1],
     [0, 1, 1, 0]
    ],
    [[0, 1, 0, 0],
     [1, 0, 1, 1],
     [0, 1, 0, 1],
     [0, 1, 1, 0]
    ]])

X = np.array([
    [
    [1, 0, 1],
    [0, 1, 0],
    [1, 1, 0],
    [0, 0, 1]
    ],
    [
    [1, 0, 0],
    [0, 1, 0],
    [1, 1, 0],
    [0, 0, 1]
    ]])

targets = np.array([
    [1, 0],
    [0, 1]
])

gcn = models.GCN(3, hidden_dims=[16, 16], output_dim=targets.shape[1], global_pool='max')
gcn.train(A, X, targets=targets, early_stopping=0.001)
print(gcn.measure_accuracy(A, X, targets))