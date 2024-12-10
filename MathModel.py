import numpy as np
import torch


#LQ model normal jump
class LQJump:
    def __init__(self, T, N, dim_x, dim_w, dim_n, vol, rates, jump_means, jump_sds, jump_type, common_noise):
        self.T = T
        self.N = N
        self.dt = T / N

        self.dim_x = dim_x
        self.dim_w = dim_w
        self.dim_n = dim_n

        self.vol = vol


        self.rates = rates
        self.jump_means = jump_means
        self.jump_sds = jump_sds
        self.jump_type = jump_type
        self.common_noise = common_noise


    #initial value is drawn from U(0,1)
    def initial(self, size):
        return torch.rand(size, self.dim_x)

    #drift (dim_x)
    def b(self, t, x_t, mu_t):
        return self.r * x_t

    #diffusion (dim_x x dim_d)
    def sigma(self, t, x_t, mu_t, u_t):
        #TODO
        return torch.ones(x_t.shape[0],self.dim_x, self.dim_w)

    #jump (dim_x x dim_n)
    def gamma(self, t, x_t, mu_t, u_t):
        #TODO
        return torch.ones(x_t.shape[0], self.dim_x, self.dim_n)

    #Compensator
    def compensator(self, size):
        expected_jumps = torch.tensor(self.rates)*torch.tensor(self.jump_means)
        expected_jumps = torch.repeat_interleave(expected_jumps.unsqueeze(0), size, 0)
        return expected_jumps.unsqueeze(-1)

    def forward_step(self, t, x_t, mu_t, u_t, dW, dP):
        dx = (self.b(t, x_t, mu_t) * self.dt + torch.matmul(self.sigma(t, x_t, mu_t, u_t), dW) +
              torch.matmul(self.gamma(t, x_t, mu_t, u_t), dP) -
              torch.matmul(self.gamma(t, x_t, mu_t, u_t), self.compensator(x_t.shape[0])))*self.dt
        return x_t + dx


    #Ongoing cost
    def f(self, t, x_t, mu_t, u_t):
        #TODO
        return - mu_t

    #Terminal cost
    def g(self, x_T, mu_T, u_T):
        #TODO
        return x_T

    #Loss function
    def loss(self, ongoing_cost, terminal_cost):
        return torch.mean(ongoing_cost + terminal_cost, dim=0)



class IBContinuous:
    def __init__(self, T, N, dim_x, dim_w, dim_n, reversion, vol, rho, q, epsilon, c,
                 rates, jump_means, jump_sds, jump_type, common_noise):
        self.T = T
        self.N = N
        self.dt = T / N

        self.dim_x = dim_x
        self.dim_w = dim_w
        self.dim_n = dim_n

        self.reversion = reversion
        self.vol = vol
        self.rho = rho
        self.q = q
        self.epsilon = epsilon
        self.c = c


        self.rates = rates
        self.jump_means = jump_means
        self.jump_sds = jump_sds
        self.jump_type = jump_type
        self.common_noise = common_noise


    #initial value is drawn from U(0,1)
    def initial(self, size):
        return torch.rand(size, self.dim_x)

    #drift (dim_x)
    def b(self, t, x_t, mu_t, u_t):
        return self.reversion * (mu_t - x_t) + u_t

    #diffusion (dim_x x dim_d)
    def sigma(self, t, x_t, mu_t, u_t):
        #TODO
        return torch.ones(x_t.shape[0],self.dim_x, self.dim_w)

    #jump (dim_x x dim_n)
    def gamma(self, t, x_t, mu_t, u_t):
        #TODO
        return torch.ones(x_t.shape[0], self.dim_x, self.dim_n)

    #Compensator
    def compensator(self, size):
        expected_jumps = torch.tensor(self.rates)*torch.tensor(self.jump_means)
        expected_jumps = torch.repeat_interleave(expected_jumps.unsqueeze(0), size, 0)
        return expected_jumps.unsqueeze(-1)

    def forward_step(self, t, x_t, mu_t, u_t, dW, dP):
        dx = (self.b(t, x_t, mu_t) * self.dt + torch.matmul(self.sigma(t, x_t, mu_t, u_t), dW) +
              torch.matmul(self.gamma(t, x_t, mu_t, u_t), dP) -
              torch.matmul(self.gamma(t, x_t, mu_t, u_t), self.compensator(x_t.shape[0])))*self.dt
        return x_t + dx


    #Ongoing cost
    def f(self, t, x_t, mu_t, u_t):
        #TODO
        return - mu_t

    #Terminal cost
    def g(self, x_T, mu_T, u_T):
        #TODO
        return x_T

    #Loss function
    def loss(self, ongoing_cost, terminal_cost):
        return torch.mean(ongoing_cost + terminal_cost, dim=0)




















