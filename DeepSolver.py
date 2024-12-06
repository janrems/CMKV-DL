import torch
import numpy as np
import matplotlib.pyplot as plt
class DeepSolver:
    def __init__(self, math_model, net, batch_size, eval_size, learning_rate, n_epochs):
        self.math_model = math_model
        self.batch_size = batch_size
        self.eval_size = eval_size
        self.n_epochs = n_epochs
        self.net = net

        self.loss_function = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(net.parameters(), learning_rate)
        self.loss_list = []
        self.y0_list = []

    def gen_brownian(self, n_samples):
        return torch.randn(n_samples, self.math_model.dim_d, 1, self.math_model.N)*np.sqrt(self.math_model.dt)


    def gen_forward(self, dW_f):
        x = torch.zeros(dW_f.shape[0], self.math_model.dim_x, self.math_model.N_ext)
        for i in range(self.math_model.N_ext):
            if i <= self.math_model.N_delta:
                t = (i - self.math_model.N_delta)*self.math_model.dt
                x[:, :, i] = self.math_model.phi(t, x[:, :, 0])
            else:
                t = (i - self.math_model.N_delta - 1)*self.math_model.dt
                drift = self.math_model.b(t, x[:, :, i - 1], x[:, :, i - 1 - self.math_model.N_delta])
                vol = torch.matmul(self.math_model.sigma(t, x[:, :, i - 1], x[:, :, i - 1 - self.math_model.N_delta]),
                                               dW_f[:, :, :, i - 1 - self.math_model.N_delta])
                x[:, :, i] = x[:, :, i - 1] + drift*self.math_model.dt + vol.view(vol.shape[:-1])

        return x

    def train(self):
        self.net.train()

        for i in range(self.n_epochs):

            dW_f = self.gen_brownian(self.batch_size)
            x = self.gen_forward(dW_f)

            y_T, y, z = self.net(x, dW_f)

            loss = self.loss_function(y_T, self.math_model.g(x[:, :, -1]))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.loss_list.append(float(loss))
            y_p = y[0, :, :self.math_model.N_delta + 1].detach().numpy()
            self.y0_list.append(y_p[0, -1])
            if i % 50 == 0:
                print("Epoch: " + str(i) + ", Loss: " + str(float(loss)))

    def eval(self):
        self.net.eval()

        dW_f = self.gen_brownian(self.eval_size)
        x = self.gen_forward(dW_f)

        y_T, y, z = self.net(x, dW_f)

        loss = self.loss_function(y_T, self.math_model.g(x[:, :, -1]))

        return y, z, x, dW_f, loss


    def plot_lists(self):

        plt.plot(self.loss_list)
        plt.show()

        plt.plot(self.y0_list)
        plt.axhline(y = self.math_model.y0_true())
        plt.show()

    def plot_realisations(self):
        t_p = torch.linspace(-self.math_model.N_delta * self.math_model.dt, 0, self.math_model.N_delta + 1)
        t_f = torch.linspace(0, self.math_model.T, self.math_model.N)
        t = torch.cat((t_p, t_f), dim=-1)

        y, z, x, dW_f, loss = self.eval()
        i = np.random.randint(y.shape[0])
        #yplt.plot(t, x[i, 0, :].detach().numpy())
        plt.plot(t, y[i, 0, :].detach().numpy())
        plt.plot(0, self.math_model.y0_true(), 'ro')
        plt.plot(self.math_model.T, self.math_model.g(x[i, :, -1]).detach().numpy(), 'ro')
        plt.show()

    def plot_realisations_f(self):
        t_f = torch.linspace(0, self.math_model.T, self.math_model.N + 1)
        y, z, x, dW_f, loss = self.eval()
        i = np.random.randint(y.shape[0])
        #plt.plot(t_f, x[i, 0, self.math_model.N_delta:].detach().numpy())
        plt.plot(t_f, y[i, 0, self.math_model.N_delta:].detach().numpy())
        plt.plot(0, self.math_model.y0_true(), 'ro')
        plt.plot(self.math_model.T, self.math_model.g(x[i, 0, -1]).detach().numpy(), 'ro')
        plt.show()



