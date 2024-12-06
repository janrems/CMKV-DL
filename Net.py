import torch
import numpy as np
import torch.nn as nn
from sklearn.linear_model import Ridge
import signatory


class Common_Noise(nn.Module):
    def __init__(self, math_model, ridge_param, depth):
        super(Common_Noise, self).__init__()

        self.math_model = math_model
        self.ridge_param = ridge_param
        self.depth = depth
        self.augment = signatory.Augment(1,
                            layer_sizes=(),
                            kernel_size=1,
                            include_time=True)

        #Layers
        self.linear1 = nn.Linear(2*self.math_model.dim_x + 1, 64)
        self.linear2 = nn.Linear(64, 32)
        self.linear3 = nn.Linear(32, self.math_model.dim_x)





    #Feed_forward pass
    def one_pass(self, inpt):
        out = torch.relu(self.linear1(inpt))
        out = torch.relu(self.linear2(out))
        return self.linear3(out)

    def conditional_expectation(self, x_t, rough, i):


        batch= x_t.shape[0]

        ridge = Ridge(alpha=self.ridge_param, tol=1e-6)
        label = x_t.detach()
        if i == 0:
            data = torch.cat([torch.zeros(batch, 1),
                              torch.zeros(batch, 1)],
                             dim=1)

            ridge.fit(data.numpy(), label.numpy())
            l = torch.tensor(ridge.coef_).view(-1, 1)
            # i=1

            return torch.matmul(torch.zeros(batch, 2), l) + ridge.intercept_
        if i == 1:
            data = torch.cat([rough.path[0][:, :2, 0],
                              torch.ones(batch, 1) * (i / self.math_model.N)], dim=1)
            ridge.fit(data.numpy(), label.numpy())

        else:
            data = rough.signature(end=i).cpu().detach()

            ridge.fit(data.numpy(), label.numpy())

        return torch.from_numpy(ridge.predict(data.numpy()))

    def forward(self,  dW, dP, common_noise):

        control_seq = torch.empty((self.batch_size, self.N, self.math_model.dim_x))

        state_seq = torch.empty((self.batch_size, self.N, self.math_model.dim_x))

        cex_seq = torch.empty((self.batch_size, self.N, self.math_model.dim_x))

        rough_path = signatory.Path(self.augment(common_noise), self.depth, basepoint=False)

        x_t = self.math_model.initial(self.batch_size)

        state_seq[:, 0, :] = x_t

        ongoing_cost = torch.zeros(self.batch_size)

        for i in range(self.math_model.N):
            t = self.math_model.dt * i

            mu_t = self.conditional_expectation(x_t, rough_path, i)
            cex_seq[:, i, :] = mu_t
            inpt = torch.cat((x_t, mu_t), 1)
            inpt = torch.cat((inpt, torch.ones(x_t.size()[0], 1) * t), 1).float()
            u_t = self.one_pass(inpt)
            control_seq[:, i, :] = u_t
            ongoing_cost += self.math_model.f(t, x_t, mu_t, u_t)*self.math_model.dt

            if i < self.math_model.N - 1:
                x_t = self.math_model.forward_step(t, x_t, mu_t, out, dW[:, i, :, :], dP[:, i, :,, :])

                state_seq[:, i + 1, :] = x_t

        terminal_cost = self.math_model.g(x_t, mu_t, u_t)

        loss = torch.mean(ongoing_cost + terminal_cost, dim=0)

        return state_seq, control_seq, cex_seq, loss





