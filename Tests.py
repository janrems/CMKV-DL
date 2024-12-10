import numpy as np
import torch
import matplotlib.pyplot as plt

from MathModel import IBContinuous
from MathModel import IBJump
from MathModel import IBCommonJump
from DeepSolver import DeepSolver


ib_c = IBContinuous(T=1.0, N=100, dim_x=1, dim_w=2, dim_n=1, reversion=1.0, vol=0.2, rho=0.2, q=1.0, epsilon=1.5,
                    c=1.0, rates=[0.0], jump_means=[0.0], jump_sds=[0.0], jump_type="None", common_noise="Brownian",
                    path="C:/Users/jan1r/Documents/Faks/Doktorat/CommonNoise/Graphs/NewGraphs/ib_c/")

ib_j = IBJump(T=1.0, N=100, dim_x=1, dim_w=2, dim_n=1, reversion=1.0, vol=0.2, rho=0.2, q=1.0, epsilon=1.5,
                    c=1.0, rates=[5.0], jump_means=[0.0], jump_sds=[0.05], jump_type="Normal", common_noise="Brownian",
                    path="C:/Users/jan1r/Documents/Faks/Doktorat/CommonNoise/Graphs/NewGraphs/ib_j/")

ib_cj = IBCommonJump(T=1.0, N=100, dim_x=1, dim_w=1, dim_n=2, reversion=1.0, vol=0.2, rho=0.2, q=1.0, epsilon=1.5,
                    c=1.0, rates=[5.0, 5.0], jump_means=[0.0, 0.0], jump_sds=[0.05, 0.05], jump_type="Normal", common_noise="Poisson",
                    path="C:/Users/jan1r/Documents/Faks/Doktorat/CommonNoise/Graphs/NewGraphs/ib_cj/")


solver = DeepSolver(math_model=ib_cj, batch_size=2**10, eval_size=2**10, learning_rate=0.0005, n_epochs=4, ridge_param=0.1, depth=2)

solver.train()

state_seq, control_seq, cex_seq, loss, dW, dP, common_noise = solver.eval()


state_seq_true, control_seq_true, cex_seq_true, loss_true = ib_j.explicit_solution(dW, dP, common_noise, state_seq)

solver.plot_loss(loss_true)

solver.plot_realisations(state_seq, control_seq, cex_seq, state_seq_true, control_seq_true, cex_seq_true)

l2state, l2control, l2cex = solver.L2(state_seq, state_seq_true, control_seq, control_seq_true, cex_seq, cex_seq_true)













