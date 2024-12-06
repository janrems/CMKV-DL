import torch
import numpy as np
import torch.nn as nn


class NetFixed(nn.Module):
    def __init__(self, math_model, dim_h, n_hidden_layers):
        super(NetFixed, self).__init__()

        self.math_model = math_model

        #Layers
        self.input_bn = nn.BatchNorm1d(2 * math_model.dim_x + 1)
        self.input_layer = nn.Linear(2 * math_model.dim_x + 1, dim_h)
        self.layer_list = nn.ModuleList([nn.Linear(dim_h, dim_h) for i in range(n_hidden_layers)])
        self.bn_list = nn.ModuleList([nn.BatchNorm1d(dim_h) for i in range(n_hidden_layers)])
        self.output_bn = nn.BatchNorm1d(dim_h)
        self.output_layer = nn.Linear(dim_h, math_model.dim_y * math_model.dim_d)
        self.activation = torch.tanh

        #Backward initial function (From t=-N_delta to t=0
        self.initial_y = nn.Parameter(torch.rand(math_model.dim_y, 1), requires_grad=True)


    #Feed_forward pass
    def one_pass(self, inpt):
        state = self.input_bn(inpt)
        state = self.input_layer(state)
        state = self.activation(state)
        for i in range(len(self.layer_list)):
            state = self.bn_list[i](state)
            state = self.layer_list[i](state)
            state = self.activation(state)
        state = self.output_bn(state)
        state = self.output_layer(state)
        return state

    def forward(self, x, dW_f):

        y_f = torch.zeros(x.shape[0], self.math_model.dim_y, self.math_model.N)
        z_f = torch.zeros(x.shape[0], self.math_model.dim_y, self.math_model.dim_d, self.math_model.N)
        y_p = torch.ones(x.shape[0], self.math_model.dim_y, 1) * self.initial_y
        y_p = torch.repeat_interleave(y_p, self.math_model.N_delta + 1, dim=-1)
        z_p = torch.zeros(x.shape[0], self.math_model.dim_y, self.math_model.dim_d, self.math_model.N_delta + 1)
        y = torch.cat((y_p, y_f), dim=-1)
        z = torch.cat((z_p, z_f), dim=-1)

        y_t = y_p[:,:,-1]
        for i in range(self.math_model.N):
            i_t = i + self.math_model.N_delta #index correspodning to current time.
            i_delta = i #index corresponding to N_delta shifted past time
            t = i * self.math_model.dt #current time

            inpt = torch.cat((x[:, :, i_t], x[:, :, i_delta], torch.ones(x.size()[0],1) * t), -1)
            z_t = self.one_pass(inpt).view(-1, self.math_model.dim_y, self.math_model.dim_d)

            driver = self.math_model.f(t, x[:, :, i_t], x[:, :, i_t], y_t, y[:, :, i_delta], z_t, z[:,:,:,i_delta])
            vol = torch.matmul(z_t, dW_f[:, :, :, i_delta])

            y_t = y_t - driver*self.math_model.dt + vol.view(vol.shape[:-1])

            #Store current values
            y[:, :, i_t + 1] = y_t
            z[:, :, :, i_t + 1] = z_t

        return y_t, y, z



