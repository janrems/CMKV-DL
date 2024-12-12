import numpy as np
import torch
import matplotlib.pyplot as plt

from MathModel import IBContinuous
from MathModel import IBJump
from MathModel import IBCommonJump
from MathModel import LQJump
from MathModel import LQCommonJump

from DeepSolver import DeepSolver




ib_c = IBContinuous(T=1.0, N=100, dim_x=1, dim_w=2, dim_n=1, reversion=1.0, vol=0.2, rho=0.2, q=1.0, epsilon=1.5,
                    c=1.0, rates=[0.0], jump_means=[0.0], jump_sds=[0.0], jump_type="None", common_noise="Brownian",
                    path="C:/Users/jan1r/Documents/Faks/Doktorat/CommonNoiseCode/CommonNoise/Graphs/ib_c/")

ib_j = IBJump(T=1.0, N=100, dim_x=1, dim_w=2, dim_n=1, reversion=1.0, vol=0.2, rho=0.2, q=1.0, epsilon=1.5,
                    c=1.0, rates=[5.0], jump_means=[0.0], jump_sds=[0.05], jump_type="Normal", common_noise="Brownian",
                    path="C:/Users/jan1r/Documents/Faks/Doktorat/CommonNoiseCode/CommonNoise/Graphs/ib_j/")

ib_cj = IBCommonJump(T=1.0, N=100, dim_x=1, dim_w=1, dim_n=2, reversion=1.0, vol=0.2, rho=0.2, q=1.0, epsilon=1.5,
                    c=1.0, rates=[5.0, 5.0], jump_means=[0.0, 0.0], jump_sds=[0.05, 0.05], jump_type="Normal",
                    common_noise="Poisson",
                    path="C:/Users/jan1r/Documents/Faks/Doktorat/CommonNoiseCode/CommonNoise/Graphs/ib_cj/")


lq_j = LQJump(T=1.0, N=100, dim_x=3, dim_w=4, dim_n=1, theta=[0.3, 0.3, 0.2], vol=[0.2, 0.1, 0.3], rates=[5.0], jump_means=[0.0],
              jump_sds=[0.05], jump_type="Normal", common_noise="Brownian",
              path="C:/Users/jan1r/Documents/Faks/Doktorat/CommonNoiseCode/CommonNoise/Graphs/lq_j/")

lq_cj = LQCommonJump(T=1.0, N=100, dim_x=1, dim_w=1, dim_n=2, rates=[5.0, 5.0], jump_means=[0.0, 0.0],
              jump_sds=[0.05, 0.05], jump_type="Normal", common_noise="Poisson",
              path="C:/Users/jan1r/Documents/Faks/Doktorat/CommonNoiseCode/CommonNoise/Graphs/lq_cj/")



solver = DeepSolver(math_model=lq_j, batch_size=2**10, eval_size=2**10, learning_rate=0.0005, n_epochs=400, ridge_param=0.1, depth=2)

solver.train()

state_seq, control_seq, cex_seq, loss, dW, dP, common_noise = solver.eval()


state_seq_true, control_seq_true, cex_seq_true, loss_true = lq_j.explicit_solution(dW, dP, common_noise, state_seq)

solver.plot_loss(loss_true)

solver.plot_realisations(state_seq, control_seq, cex_seq, state_seq_true, control_seq_true, cex_seq_true)

l2state, l2control, l2cex = solver.L2(state_seq, state_seq_true, control_seq, control_seq_true, cex_seq, cex_seq_true)


d = 3
w = 4
bs = 10
mu_t = torch.ones(bs, d)
mu_t[0, :] = mu_t[0, :] * 2
mu_t[:,-1] = mu_t[:,-1] * 3

theta = 1.0
sig = [0.1]

tt = torch.ones(bs,d,1)*theta

sigs = torch.tensor(sig).float().unsqueeze(0)
sigs = torch.repeat_interleave(sigs, bs, 0)

sm = sigs*mu_t
diag = torch.diag_embed(sm,0)

out = torch.cat((tt, diag), dim=-1)
out[1,:,:]

a = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).float()
b = torch.diag_embed(a,0)


thetas = torch.tensor([2,3,4], dtype=torch.float32).unsqueeze(-1)
thetas = torch.repeat_interleave(thetas.unsqueeze(0), 5, 0)

thetas.shape
thetas[0,:,:]





