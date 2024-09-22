import numpy as np
import models
A = np.array([
        [0, 1, 0, 0],
        [1, 0, 1, 1],
        [0, 1, 0, 1],
        [0, 1, 1, 0]
    ])

X = np.array([
    [1, 0, 1],
    [0, 1, 0],
    [1, 1, 0],
    [0, 0, 1]
])

targets = np.array([
    [1, 0],
    [0, 1],
    [1, 0],
    [0, 1]
])

gcn = models.GCN(A, X, hidden_dims=[16, 16], output_dim=targets.shape[1])
gcn.train(targets=targets, early_stopping=0.1)