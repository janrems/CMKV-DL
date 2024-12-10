import torch
import numpy as np

class DataGenerator:
    def __init__(self, math_model):
        self.math_model = math_model

    def brownian_increments(self, size):
        dW = torch.randn(size//2, self.math_model.N, self.math_model.dim_w, 1) * np.sqrt(self.math_model.dt)
        dW = torch.cat((dW, -dW), 0)
        return dW

    def jumps(self, size):
        mus_t = torch.tensor(self.math_model.jump_means).float().unsqueeze(0)
        mus_t = torch.repeat_interleave(mus_t, size, 0)
        mus_t = mus_t.view(-1).unsqueeze(0)
        mus_t = torch.repeat_interleave(mus_t, self.math_model.N, 0)

        sigmas_t = torch.tensor(self.math_model.jump_sds).float().unsqueeze(0)
        sigmas_t = torch.repeat_interleave(sigmas_t, size, 0)
        sigmas_t = sigmas_t.view(-1).unsqueeze(0)
        sigmas_t = torch.repeat_interleave(sigmas_t, self.math_model.N, 0)

        if self.math_model.jump_type == "Normal":
            return torch.normal(mus_t, sigmas_t)
        else:
            raise ValueError("Jump type not implemented")

    def jump_times(self, size):

        rates_t = torch.tensor(self.math_model.rates).float().unsqueeze(0)
        rates_t = torch.repeat_interleave(rates_t, size, 0)

        poiss = torch.poisson(rates_t * self.math_model.T)
        poiss = poiss.view(-1)


        max_length = int(poiss.max())
        bs2 = int(poiss.size()[0])

        tj = torch.randint(self.math_model.N, (bs2, max_length))
        range_tensor = torch.arange(max_length)
        bool_tensor = range_tensor >= poiss.view(-1, 1)
        tj[bool_tensor] = self.math_model.N

        return tj

    def compound_poisson_increments(self, size):
        jumps = self.jumps(size)
        jump_times = self.jump_times(size)
        jumps = torch.cat((jumps, torch.ones(1, size * self.math_model.dim_n)), dim=0)
        dP = torch.zeros(self.math_model.N + 1, size * self.math_model.dim_n).scatter_(0, jump_times.transpose(0, 1), jumps,
                                                                      reduce="add")[:-1, :]
        dP = dP.reshape(self.math_model.N, size, self.math_model.dim_n, 1)
        return torch.transpose(dP, 0, 1)

    def process_from_increments(self, increments):
        process = torch.cumsum(increments, dim=1)
        process = torch.roll(process, 1, 1)
        new_size = torch.Size([process.shape[0]] + list(process.shape[2:]))
        process[:, 0, :] = torch.zeros(new_size)

        return process


