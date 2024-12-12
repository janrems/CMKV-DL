import numpy as np
import torch
from scipy.integrate import odeint



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
        sigma = torch.tensor([self.vol * self.rho, self.vol * np.sqrt(1 - self.rho ** 2)], dtype=torch.float32)
        sigma = torch.repeat_interleave(sigma.unsqueeze(0), self.dim_x, 0)
        sigma = torch.repeat_interleave(sigma.unsqueeze(0), x_t.shape[0], 0)
        return sigma

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

        ongoing_cost = self.f(0, x_t, mu_t, u_t) * self.dt

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
        sigma = torch.tensor([self.vol * self.rho, self.vol * np.sqrt(1 - self.rho ** 2)], dtype=torch.float32)
        sigma = torch.repeat_interleave(sigma.unsqueeze(0), self.dim_x, 0)
        sigma = torch.repeat_interleave(sigma.unsqueeze(0), x_t.shape[0], 0)
        return sigma

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

        ongoing_cost = self.f(0, x_t, mu_t, u_t)*self.dt

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
        gamma = torch.tensor([self.rho, 1 - self.rho], dtype=torch.float32)
        gamma = torch.repeat_interleave(gamma.unsqueeze(0), self.dim_x, 0)
        gamma = torch.repeat_interleave(gamma.unsqueeze(0), x_t.shape[0], 0)
        return gamma

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


class LQJump:
    def __init__(self, T, N, dim_x, dim_w, dim_n, theta, vol, rates,
                 jump_means, jump_sds, jump_type, common_noise, path):
        self.T = T
        self.N = N
        self.dt = T / N

        self.dim_x = dim_x
        self.dim_w = dim_w
        self.dim_n = dim_n

        self.theta = theta
        self.vol = vol

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
        return u_t

    #diffusion (dim_x x dim_d)
    def sigma(self, t, x_t, mu_t, u_t):
        thetas = torch.ones(x_t.shape[0], self.dim_x, 1) * self.theta
        vols = torch.tensor(self.vol, dtype=torch.float32).unsqueeze(0)
        vols = torch.repeat_interleave(vols, x_t.shape[0], 0)
        vols = vols * mu_t
        diag = torch.diag_embed(vols, 0)

        out = torch.cat((thetas, diag), dim=-1)

        return out


    # jump (dim_x x dim_n)
    def gamma(self, t, x_t, mu_t, u_t):
        return mu_t.unsqueeze(-1)

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
        return 0.5 * torch.linalg.norm(u_t, dim=-1) ** 2


    #Terminal cost
    def g(self, x_T, mu_T):
        return 0.5*torch.linalg.norm(x_T, dim=-1) ** 2

    #Loss function
    def loss(self, ongoing_cost, terminal_cost):
        return torch.mean(ongoing_cost + terminal_cost, dim=0)

    def explicit_solution(self, dW, dP, common_noise, state_seq):
        if self.dim_x != 1:
            raise ValueError("Explicit solution only implemented for dim_x = 1")

        def riccati_sys(y, t, theta, vol, mu, sd, l):
            k0, k1, k2, k3 = y
            dyt = [-theta ** 2 * (k1 + k2 + k3), -0.5 * (2 * k1 + k2) ** 2, -2 * (2 * k1 + k2) * (k2 + 2 * k3),
                   -0.5 * (k2 + 2 * k3) ** 2 - k1 * (vol ** 2 + l * (mu ** 2 + sd ** 2))]
            return dyt

        size = dW.shape[0]
        yT = [0.0, -0.5, 0.0, 0.0]
        tb = np.linspace(self.T, 0, self.N)

        sol = odeint(riccati_sys, yT, tb, args=(self.theta, self.vol, self.jump_means[0],
                                                self.jump_sds[0], self.rates[0]))
        sol = np.flip(sol, 0)
        sol = sol.T

        sol = np.expand_dims(sol, axis=1)
        sol = np.repeat(sol, size, axis=1)
        sol = np.expand_dims(sol, axis=-1)

        k0, k1, k2, k3 = torch.tensor(sol, dtype=torch.float32)
        k = 2 * (k1 + k2 + k3)

        state_seq_true = state_seq.clone()
        x_0 = state_seq_true[:, 0, :]

        ex_0 = torch.mean(x_0[:, 0], dim=0)
        cex_seq_true = torch.ones(size, self.N, self.dim_x) * ex_0

        control_seq_true = (2 * k1 + k2) * state_seq_true + (k2 + 2 * k3) * cex_seq_true

        u_t = control_seq_true[:, 0, :]
        x_t = x_0
        mu_t = cex_seq_true[:, 0, :]

        ongoing_cost = self.f(0, x_t, mu_t, u_t) * self.dt

        for i in range(self.N - 1):
            t = self.dt * i
            x_t = self.forward_step(t, x_t, mu_t, u_t, dW[:, i, :, :], dP[:, i, :, :])
            mu_t = mu_t + k[:, i, :]*mu_t*self.dt + self.theta*(common_noise[:, i + 1, :] - common_noise[:, i, :])
            u_t = (2 * k1 + k2)[:, i + 1, :] * x_t + (k2 + 2 * k3)[:, i + 1, :] * mu_t
            ongoing_cost += self.f(t, x_t, mu_t, u_t) * self.dt

            state_seq_true[:, i + 1, :] = x_t
            control_seq_true[:, i + 1, :] = u_t
            cex_seq_true[:, i + 1, :] = mu_t

        terminal_cost = self.g(x_t, mu_t)
        loss_true = self.loss(ongoing_cost, terminal_cost)

        return state_seq_true, control_seq_true, cex_seq_true, loss_true


class LQCommonJump:
    def __init__(self, T, N, dim_x, dim_w, dim_n, rates, jump_means, jump_sds, jump_type, common_noise, path):
        self.T = T
        self.N = N
        self.dt = T / N

        self.dim_x = dim_x
        self.dim_w = dim_w
        self.dim_n = dim_n

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
        return u_t




    # jump (dim_x x dim_n)
    def gamma(self, t, x_t, mu_t, u_t):
        ones = torch.ones(x_t.shape[0], self.dim_x, 1)
        gamma = torch.cat((ones, mu_t.unsqueeze(-1)), dim=2)
        return gamma

    # Compensator
    def compensator(self, size):
        expected_jumps = torch.tensor(self.rates) * torch.tensor(self.jump_means)
        expected_jumps = torch.repeat_interleave(expected_jumps.unsqueeze(0), size, 0)
        return expected_jumps.unsqueeze(-1)

    def forward_step(self, t, x_t, mu_t, u_t, dW, dP):
        dx = (self.b(t, x_t, mu_t, u_t) * self.dt +
              torch.matmul(self.gamma(t, x_t, mu_t, u_t), dP).view(x_t.shape[0], self.dim_x) -
              torch.matmul(self.gamma(t, x_t, mu_t, u_t),
                           self.compensator(x_t.shape[0])).view(x_t.shape[0], self.dim_x) * self.dt)
        return x_t + dx

    #Ongoing cost
    def f(self, t, x_t, mu_t, u_t):
        return 0.5 * torch.linalg.norm(u_t, dim=-1) ** 2


    #Terminal cost
    def g(self, x_T, mu_T):
        return 0.5*torch.linalg.norm(x_T, dim=-1) ** 2

    #Loss function
    def loss(self, ongoing_cost, terminal_cost):
        return torch.mean(ongoing_cost + terminal_cost, dim=0)

    def explicit_solution(self, dW, dP, common_noise, state_seq):
        if self.dim_x != 1:
            raise ValueError("Explicit solution only implemented for dim_x = 1")

        def riccati_sys(y, t, mu, sd, l):
            k0, k1, k2, k3, k4, k5 = y
            dyt = [-(0.5 * (k4 + k5) ** 2 + (k1 + k3) * l * (mu ** 2 + sd ** 2) + (k5 - k2) * l * mu + k1 * l * (
                        mu ** 2 + sd ** 2)), -0.5 * (2 * k1 + k2) ** 2, - (2 * k1 + k2) * (k2 + 2 * k3),
                   -0.5 * (k2 + 2 * k3) ** 2, - ((2 * k1 + k2) * (k4 + k5) + k2 * l * mu),
                   - ((k2 + 2 * k3) * (k4 + k5) + (2 * k3 - k2) * l * mu - k2 * l * mu)]
            return dyt

        size = dW.shape[0]
        yT = [0.0, -0.5, 0.0, 0.0,0.0,0.0]
        tb = np.linspace(self.T, 0, self.N)

        sol = odeint(riccati_sys, yT, tb, args=(self.jump_means[0],
                                                self.jump_sds[0], self.rates[0]))
        sol = np.flip(sol, 0)
        sol = sol.T

        sol = np.expand_dims(sol, axis=1)
        sol = np.repeat(sol, size, axis=1)
        sol = np.expand_dims(sol, axis=-1)

        k0, k1, k2, k3, k4, k5 = torch.tensor(sol, dtype=torch.float32)
        k = 2 * (k1 + k2 + k3)



        state_seq_true = state_seq.clone()
        x_0 = state_seq_true[:, 0, :]

        ex_0 = torch.mean(x_0[:, 0], dim=0)
        cex_seq_true = torch.ones(size, self.N, self.dim_x) * ex_0

        control_seq_true = (2 * k1 + k2) * state_seq_true + (k2 + 2 * k3) * cex_seq_true + k4 + k5

        u_t = control_seq_true[:, 0, :]
        x_t = x_0
        mu_t = cex_seq_true[:, 0, :]

        ongoing_cost = self.f(0, x_t, mu_t, u_t) * self.dt

        for i in range(self.N - 1):
            t = self.dt * i
            x_t = self.forward_step(t, x_t, mu_t, u_t, dW[:, i, :, :], dP[:, i, :, :])
            mu_t = mu_t + (k[:, i, :]*mu_t + k4[:, i, :] + k5[:, i, :])*self.dt + (common_noise[:, i + 1, :] - common_noise[:, i, :])
            u_t = (2 * k1 + k2)[:, i + 1, :] * x_t + (k2 + 2 * k3)[:, i + 1, :] * mu_t + k4[:, i + 1, :] + k5[:, i + 1, :]
            ongoing_cost += self.f(t, x_t, mu_t, u_t) * self.dt

            state_seq_true[:, i + 1, :] = x_t
            control_seq_true[:, i + 1, :] = u_t
            cex_seq_true[:, i + 1, :] = mu_t

        terminal_cost = self.g(x_t, mu_t)
        loss_true = self.loss(ongoing_cost, terminal_cost)

        return state_seq_true, control_seq_true, cex_seq_true, loss_true
