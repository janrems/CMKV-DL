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
        return torch.ones(x_t.shape[0], self.dim_x, self.dim_w)

    #jump (dim_x x dim_n)
    def gamma(self, t, x_t, mu_t, u_t):
        #TODO
        return torch.ones(x_t.shape[0], self.dim_x, self.dim_n)

    #Compensator
    def compensator(self, size):
        expected_jumps = torch.tensor(self.rates) * torch.tensor(self.jump_means)
        expected_jumps = torch.repeat_interleave(expected_jumps.unsqueeze(0), size, 0)
        return expected_jumps.unsqueeze(-1)

    def forward_step(self, t, x_t, mu_t, u_t, dW, dP):
        dx = (self.b(t, x_t, mu_t) * self.dt + torch.matmul(self.sigma(t, x_t, mu_t, u_t), dW) +
              torch.matmul(self.gamma(t, x_t, mu_t, u_t), dP) -
              torch.matmul(self.gamma(t, x_t, mu_t, u_t), self.compensator(x_t.shape[0]))) * self.dt
        return x_t + dx

    #Ongoing cost
    def f(self, t, x_t, mu_t, u_t):
        #TODO
        return - mu_t

    #Terminal cost
    def g(self, x_T, mu_T):
        #TODO
        return x_T

    #Loss function
    def loss(self, ongoing_cost, terminal_cost):
        return torch.mean(ongoing_cost + terminal_cost, dim=0)


class IBContinuous:
    def __init__(self, T, N, dim_x, dim_w, dim_n, reversion, vol, rho, q, epsilon, c,
                 rates, jump_means, jump_sds, jump_type, common_noise, path):
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

        self.path = path

    #initial value is drawn from U(0,1)
    def initial(self, size):
        return torch.rand(size, self.dim_x)

    #drift (dim_x)
    def b(self, t, x_t, mu_t, u_t):
        return self.reversion * (mu_t - x_t) + u_t

    #diffusion (dim_x x dim_d)
    def sigma(self, t, x_t, mu_t, u_t):
        sigma = torch.tensor([self.vol * self.rho, self.vol * np.sqrt(1 - self.rho ** 2)])
        sigma = torch.repeat_interleave(sigma.unsqueeze(0), self.dim_x, 0)
        sigma = torch.repeat_interleave(sigma.unsqueeze(0), x_t.shape[0], 0)
        return sigma.float()

    def forward_step(self, t, x_t, mu_t, u_t, dW, dP):
        dx = self.b(t, x_t, mu_t, u_t) * self.dt + torch.matmul(self.sigma(t, x_t, mu_t, u_t), dW).view(x_t.shape[0],
                                                                                                        self.dim_x)
        return x_t + dx

    #Ongoing cost
    def f(self, t, x_t, mu_t, u_t):
        f1 = 0.5 * torch.linalg.norm(u_t, dim=-1) ** 2
        f2 = self.q * torch.bmm(u_t.view(x_t.shape[0], 1, self.dim_x), (mu_t - x_t).view(x_t.shape[0], self.dim_x, 1))
        f3 = 0.5 * self.epsilon * torch.linalg.norm(mu_t - x_t, dim=-1) ** 2
        return f1 - f2.squeeze() + f3

    #Terminal cost
    def g(self, x_T, mu_T):
        return 0.5 * self.c * torch.linalg.norm(x_T - mu_T, dim=-1) ** 2

    #Loss function
    def loss(self, ongoing_cost, terminal_cost):
        return torch.mean(ongoing_cost + terminal_cost, dim=0)

    def explicit_solution(self, dW, dP, common_noise, state_seq):
        if self.dim_x != 1:
            raise ValueError("Explicit solution only implemented for dim_x = 1")

        t = torch.linspace(0, self.T, self.N)
        time = torch.unsqueeze(t, dim=1)
        time = torch.unsqueeze(time, dim=0)
        time = torch.repeat_interleave(time, repeats=dW.shape[0], dim=0)

        delta_p = -(self.reversion + self.q) + np.sqrt((self.reversion + self.q) ** 2 + self.epsilon - self.q ** 2)
        delta_m = -(self.reversion + self.q) - np.sqrt((self.reversion + self.q) ** 2 + self.epsilon - self.q ** 2)

        Gamma_t = (0.5 * ((self.epsilon - self.q ** 2) * (torch.exp((delta_p - delta_m) * (self.T - time)) - 1.0) +
                          self.c * (delta_p * torch.exp((delta_p - delta_m) * (self.T - time)) - delta_m)) /
                   (self.c * (torch.exp((delta_p - delta_m) * (self.T - time)) - 1.0) +
                    (-delta_m * torch.exp((delta_p - delta_m) * (self.T - time)) + delta_p)))

        state_seq_true = state_seq.clone()
        x_0 = state_seq_true[:, 0, :]

        ex_0 = torch.mean(x_0[:, 0], dim=0)
        cex_seq_true = ex_0 + self.rho * self.vol * common_noise

        control_seq_true = -(2 * Gamma_t + self.q) * (state_seq_true - cex_seq_true)

        u_t = control_seq_true[:, 0, :]
        x_t = x_0
        mu_t = cex_seq_true[:, 0, :]

        ongoing_cost = torch.zeros(x_0.shape[0])

        for i in range(self.N - 1):
            t = self.dt * i
            x_t = self.forward_step(t, x_t, mu_t, u_t, dW[:, i, :, :], dP[:, i, :, :])
            mu_t = cex_seq_true[:, i + 1, :]
            u_t = -(2 * Gamma_t[:, i + 1, :] + self.q) * (x_t - mu_t)
            ongoing_cost += self.f(t, x_t, mu_t, u_t) * self.dt

            state_seq_true[:, i + 1, :] = x_t
            control_seq_true[:, i + 1, :] = u_t

        terminal_cost = self.g(x_t, mu_t)
        loss_true = self.loss(ongoing_cost, terminal_cost)

        return state_seq_true, control_seq_true, cex_seq_true, loss_true


class IBJump:
    def __init__(self, T, N, dim_x, dim_w, dim_n, reversion, vol, rho, q, epsilon, c,
                 rates, jump_means, jump_sds, jump_type, common_noise, path):
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

        self.path = path

    #initial value is drawn from U(0,1)
    def initial(self, size):
        return torch.rand(size, self.dim_x)

    #drift (dim_x)
    def b(self, t, x_t, mu_t, u_t):
        return self.reversion * (mu_t - x_t) + u_t

    #diffusion (dim_x x dim_d)
    def sigma(self, t, x_t, mu_t, u_t):
        sigma = torch.tensor([self.vol * self.rho, self.vol * np.sqrt(1 - self.rho ** 2)])
        sigma = torch.repeat_interleave(sigma.unsqueeze(0), self.dim_x, 0)
        sigma = torch.repeat_interleave(sigma.unsqueeze(0), x_t.shape[0], 0)
        return sigma.float()

    # jump (dim_x x dim_n)
    def gamma(self, t, x_t, mu_t, u_t):
        return torch.ones(x_t.shape[0], self.dim_x, self.dim_n)

    # Compensator
    def compensator(self, size):
        expected_jumps = torch.tensor(self.rates) * torch.tensor(self.jump_means)
        expected_jumps = torch.repeat_interleave(expected_jumps.unsqueeze(0), size, 0)
        return expected_jumps.unsqueeze(-1)

    def forward_step(self, t, x_t, mu_t, u_t, dW, dP):
        dx = (self.b(t, x_t, mu_t, u_t) * self.dt +
              torch.matmul(self.sigma(t, x_t, mu_t, u_t), dW).view(x_t.shape[0], self.dim_x) +
              torch.matmul(self.gamma(t, x_t, mu_t, u_t), dP).view(x_t.shape[0], self.dim_x) -
              torch.matmul(self.gamma(t, x_t, mu_t, u_t),
                           self.compensator(x_t.shape[0])).view(x_t.shape[0], self.dim_x) * self.dt)
        return x_t + dx

    #Ongoing cost
    def f(self, t, x_t, mu_t, u_t):
        f1 = 0.5 * torch.linalg.norm(u_t, dim=-1) ** 2
        f2 = self.q * torch.bmm(u_t.view(x_t.shape[0], 1, self.dim_x), (mu_t - x_t).view(x_t.shape[0], self.dim_x, 1))
        f3 = 0.5 * self.epsilon * torch.linalg.norm(mu_t - x_t, dim=-1) ** 2
        return f1 - f2.squeeze() + f3

    #Terminal cost
    def g(self, x_T, mu_T):
        return 0.5 * self.c * torch.linalg.norm(x_T - mu_T, dim=-1) ** 2

    #Loss function
    def loss(self, ongoing_cost, terminal_cost):
        return torch.mean(ongoing_cost + terminal_cost, dim=0)

    def explicit_solution(self, dW, dP, common_noise, state_seq):
        if self.dim_x != 1:
            raise ValueError("Explicit solution only implemented for dim_x = 1")

        t = torch.linspace(0, self.T, self.N)
        time = torch.unsqueeze(t, dim=1)
        time = torch.unsqueeze(time, dim=0)
        time = torch.repeat_interleave(time, repeats=dW.shape[0], dim=0)

        delta_p = -(self.reversion + self.q) + np.sqrt((self.reversion + self.q) ** 2 + self.epsilon - self.q ** 2)
        delta_m = -(self.reversion + self.q) - np.sqrt((self.reversion + self.q) ** 2 + self.epsilon - self.q ** 2)

        Gamma_t = (0.5 * ((self.epsilon - self.q ** 2) * (torch.exp((delta_p - delta_m) * (self.T - time)) - 1.0) +
                          self.c * (delta_p * torch.exp((delta_p - delta_m) * (self.T - time)) - delta_m)) /
                   (self.c * (torch.exp((delta_p - delta_m) * (self.T - time)) - 1.0) +
                    (-delta_m * torch.exp((delta_p - delta_m) * (self.T - time)) + delta_p)))

        state_seq_true = state_seq.clone()
        x_0 = state_seq_true[:, 0, :]

        ex_0 = torch.mean(x_0[:, 0], dim=0)
        cex_seq_true = ex_0 + self.rho * self.vol * common_noise

        control_seq_true = -(2 * Gamma_t + self.q) * (state_seq_true - cex_seq_true)

        u_t = control_seq_true[:, 0, :]
        x_t = x_0
        mu_t = cex_seq_true[:, 0, :]

        ongoing_cost = torch.zeros(x_0.shape[0])

        for i in range(self.N - 1):
            t = self.dt * i
            x_t = self.forward_step(t, x_t, mu_t, u_t, dW[:, i, :, :], dP[:, i, :, :])
            mu_t = cex_seq_true[:, i + 1, :]
            u_t = -(2 * Gamma_t[:, i + 1, :] + self.q) * (x_t - mu_t)
            ongoing_cost += self.f(t, x_t, mu_t, u_t) * self.dt

            state_seq_true[:, i + 1, :] = x_t
            control_seq_true[:, i + 1, :] = u_t

        terminal_cost = self.g(x_t, mu_t)
        loss_true = self.loss(ongoing_cost, terminal_cost)

        return state_seq_true, control_seq_true, cex_seq_true, loss_true



class IBCommonJump:
    def __init__(self, T, N, dim_x, dim_w, dim_n, reversion, vol, rho, q, epsilon, c,
                 rates, jump_means, jump_sds, jump_type, common_noise, path):
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

        self.path = path

    #initial value is drawn from U(0,1)
    def initial(self, size):
        return torch.rand(size, self.dim_x)

    #drift (dim_x)
    def b(self, t, x_t, mu_t, u_t):
        return self.reversion * (mu_t - x_t) + u_t

    #diffusion (dim_x x dim_d)
    def sigma(self, t, x_t, mu_t, u_t):
        return torch.ones(x_t.shape[0], self.dim_x, self.dim_w)*self.vol

    # jump (dim_x x dim_n)
    def gamma(self, t, x_t, mu_t, u_t):
        gamma = torch.tensor([self.rho, 1 - self.rho])
        gamma = torch.repeat_interleave(gamma.unsqueeze(0), self.dim_x, 0)
        gamma = torch.repeat_interleave(gamma.unsqueeze(0), x_t.shape[0], 0)
        return gamma.float()

    # Compensator
    def compensator(self, size):
        expected_jumps = torch.tensor(self.rates) * torch.tensor(self.jump_means)
        expected_jumps = torch.repeat_interleave(expected_jumps.unsqueeze(0), size, 0)
        return expected_jumps.unsqueeze(-1)

    def forward_step(self, t, x_t, mu_t, u_t, dW, dP):
        dx = (self.b(t, x_t, mu_t, u_t) * self.dt +
              torch.matmul(self.sigma(t, x_t, mu_t, u_t), dW).view(x_t.shape[0], self.dim_x) +
              torch.matmul(self.gamma(t, x_t, mu_t, u_t), dP).view(x_t.shape[0], self.dim_x) -
              torch.matmul(self.gamma(t, x_t, mu_t, u_t),
                           self.compensator(x_t.shape[0])).view(x_t.shape[0], self.dim_x) * self.dt)
        return x_t + dx

    #Ongoing cost
    def f(self, t, x_t, mu_t, u_t):
        f1 = 0.5 * torch.linalg.norm(u_t, dim=-1) ** 2
        f2 = self.q * torch.bmm(u_t.view(x_t.shape[0], 1, self.dim_x), (mu_t - x_t).view(x_t.shape[0], self.dim_x, 1))
        f3 = 0.5 * self.epsilon * torch.linalg.norm(mu_t - x_t, dim=-1) ** 2
        return f1 - f2.squeeze() + f3

    #Terminal cost
    def g(self, x_T, mu_T):
        return 0.5 * self.c * torch.linalg.norm(x_T - mu_T, dim=-1) ** 2

    #Loss function
    def loss(self, ongoing_cost, terminal_cost):
        return torch.mean(ongoing_cost + terminal_cost, dim=0)
