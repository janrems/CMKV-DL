import numpy as np
import torch
import matplotlib.pyplot as plt


from MathModel import LookBack
from Net import LSTMModel
from DeepSolver import DeepSolver

#Lookback option model from Fang et al.
dim_x = 1
x0 = torch.ones(dim_x)*1.0

model = LookBack(1,50,1,1,1,0.05,0.2, x0)
net = LSTMModel(model, model.dim_x, 51, 3,0.2)
batch_size = 2**8
eval_size = 2**10
solver = DeepSolver(model, net, batch_size, eval_size, 0.001 , 3000)

solver.train()


solver.plot_lists()
solver.plot_realisations()

y, z, x, dW, loss = solver.eval()
y_true = model.y_true(x)
torch.mean(model.dt * torch.sum(torch.abs(y[:,:,0] - y_true),dim=0))/torch.mean(model.dt * torch.sum(torch.abs(y_true),dim=0))


import numpy as np
from sklearn.linear_model import Ridge

# Example data
X = np.random.rand(100, 10)  # 100 samples, 10 features
y = np.random.rand(100, 3)   # 100 samples, 3 targets (multivariate)

# Fit Ridge regression
ridge = Ridge(alpha=1.0)
ridge.fit(X, y)

a = ridge.coef_

# Predict
y_pred = ridge.predict(X)
print(y_pred.shape)  # Output: (100, 3)
























