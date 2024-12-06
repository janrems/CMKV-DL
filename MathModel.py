import numpy as np
import torch


#Lookback option model from Fang et al.
class LookBack:
    def __init__(self, T, N, dim_x, dim_d, dim_y, r, vol, x0):
        self.T = T
        self.N = N
        self.dim_x = dim_x
        self.dim_d = dim_d
        self.dim_y = dim_y
        self.dt = T/N
        self.r = r
        self.vol = vol
        self.x0 = x0

    #Forward drift
    def b(self, t, x_t, x):
        return self.r * x_t

    #Forward diffusion
    def sigma(self, t, x_t, x):
        sig = self.vol * x_t
        return torch.broadcast_to(sig.unsqueeze(-1),(x_t.shape[0],self.dim_x, self.dim_d))


    #Backward generator
    def f(self, t, x_t, x, y_t, z_t):
        return - self.r * y_t

    #Backward terminal. Warning: must map from dim_x to dim_y
    def g(self, x_T, x):
        x_TN = torch.broadcast_to(x_T,(self.N + 1, x_T.shape[0], self.dim_x)).clone()
        sum = torch.sum(x - x_TN, dim=-1)
        return torch.max(sum, dim=0).values

    def y_true(self, x):
        M = torch.cummax(torch.sum(x, dim=-1), dim=0).values
        M_T = M[-1,:]
        M = M[:-1,:]
        x = torch.sum(x, dim=-1)
        x_T = x[-1,:]
        x = x[:-1,:]
        t = torch.linspace(0, self.T, self.N + 1)[:-1]
        t = torch.broadcast_to(t.unsqueeze(-1), (self.N, x.shape[1]))
        tau = self.T - t

        a1 = (torch.log(x/M) + (self.r + 0.5*self.vol**2)*tau)/(self.vol*torch.sqrt(tau))
        a2 = a1 - self.vol*torch.sqrt(tau)
        a3 = a1 - 2*self.r*torch.sqrt(tau)/self.vol

        nor = torch.distributions.normal.Normal(torch.zeros(self.N, x.shape[1]),torch.ones(self.N, x.shape[1]))
        val = M*torch.exp(-self.r*tau)*nor.cdf(-a2) - x*nor.cdf(-a1) +\
                x*(self.vol**2)/(2*self.r)*(-torch.exp(-self.r*tau)*nor.cdf(a3)*(M/x)**(2*self.r/(self.vol**2)) +
                                          nor.cdf(a1))

        val_T = (M_T-x_T).unsqueeze(0)
        return torch.cat([val,val_T],dim=0)

    def y0_true(self):
        tau = torch.tensor([self.T], dtype=torch.float32)
        x0 = torch.sum(self.x0)

        a1 = (self.r + 0.5*self.vol**2)*tau/(self.vol*torch.sqrt(tau))
        a2 = a1 - self.vol*torch.sqrt(tau)
        a3 = a1 - 2*self.r*torch.sqrt(tau)/self.vol

        nor = torch.distributions.normal.Normal(0,1)
        val = x0*torch.exp(-self.r*tau)*nor.cdf(-a2) - x0*nor.cdf(-a1) +\
                x0*(self.vol**2)/(2*self.r)*(-torch.exp(-self.r*tau)*nor.cdf(a3) + nor.cdf(a1))



        return val.numpy()


















