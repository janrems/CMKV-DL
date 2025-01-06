import numpy as np
import torch
import matplotlib.pyplot as plt

from MathModel import IBContinuous
from MathModel import IBJump
from MathModel import IBCommonJump
from MathModel import LQJump
from MathModel import LQCommonJump

from DeepSolver import DeepSolver

path = "Insert_path_here/"


np.random.uniform(low=0.05, high=0.4, size=(10))

ib_c = IBContinuous(T=1.0, N=100, dim_x=1, dim_w=2, dim_n=1, reversion=1.0, vol=0.2, rho=0.2, q=1.0, epsilon=1.5,
                    c=1.0, rates=[0.0], jump_means=[0.0], jump_sds=[0.0], jump_type="None", common_noise="Brownian",
                    path=path)

ib_j = IBJump(T=1.0, N=100, dim_x=1, dim_w=2, dim_n=1, reversion=1.0, vol=0.2, rho=0.2, q=1.0, epsilon=1.5,
                    c=1.0, rates=[5.0], jump_means=[0.0], jump_sds=[0.05], jump_type="Normal", common_noise="Brownian",
                    path=path)

ib_cj = IBCommonJump(T=1.0, N=100, dim_x=1, dim_w=1, dim_n=2, reversion=1.0, vol=0.2, rho=0.2, q=1.0, epsilon=1.5,
                    c=1.0, rates=[5.0, 5.0], jump_means=[0.0, 0.0], jump_sds=[0.05, 0.05], jump_type="Normal",
                    common_noise="Poisson",
                    path=path)


lq_j = LQJump(T=1.0, N=100, dim_x=10, dim_w=11, dim_n=10, theta=np.random.uniform(low=0.05, high=0.4, size=(10)), vol=np.random.uniform(low=0.05, high=0.4, size=(10)), rates=10*[5.0], jump_means=10*[0.0],
              jump_sds=10*[0.05], jump_type="Normal", common_noise="Brownian",
              path=path)

lq_cj = LQCommonJump(T=1.0, N=100, dim_x=1, dim_w=1, dim_n=2, rates=[5.0, 5.0], jump_means=[0.0, 0.0],
              jump_sds=[0.05, 0.05], jump_type="Normal", common_noise="Poisson",
              path=path)



solver = DeepSolver(math_model=lq_j, batch_size=2**10, eval_size=2**10, learning_rate=0.0005, n_epochs=1000, ridge_param=0.1, depth=2)

solver.train()

state_seq, control_seq, cex_seq, loss, dW, dP, common_noise = solver.eval()

state_seq_true, control_seq_true, cex_seq_true, loss_true = lq_j.explicit_solution(dW, dP, common_noise, state_seq)

solver.plot_loss(loss_true,50)

solver.plot_realisations(state_seq, control_seq, cex_seq, state_seq_true, control_seq_true, cex_seq_true)

l2state, l2control, l2cex = solver.L2(state_seq, state_seq_true, control_seq, control_seq_true, cex_seq, cex_seq_true)







