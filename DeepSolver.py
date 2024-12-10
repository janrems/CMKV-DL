import torch
import numpy as np
import matplotlib.pyplot as plt

from Net import Common_Noise
from DataGenerator import DataGenerator
class DeepSolver:
    def __init__(self, math_model, batch_size, eval_size, learning_rate, n_epochs, ridge_param, depth, path):
        self.math_model = math_model
        self.generate = DataGenerator(math_model)
        self.batch_size = batch_size
        self.eval_size = eval_size
        self.n_epochs = n_epochs
        self.net = Common_Noise(math_model, ridge_param, depth)

        self.optimizer = torch.optim.Adam(self.net.parameters(), learning_rate)
        self.loss_list = []
        self.path = path

    def common_noise(self, dW, dP):
        if self.math_model.common_noise == "brownian":
            return self.generate.process_from_increments(dW[:, :, 0, :])
        elif self.math_model.common_noise == "poisson":
            return self.generate.process_from_increments(dP[:, :, 0, :])
        elif self.math_model.common_noise == "mixed":
            Wt = self.generate.process_from_increments(dW[:, :, 0, :])
            Pt = self.generate.process_from_increments(dP[:, :, 0, :])
            return torch.cat((Wt, Pt), dim=-1).unsqueeze(-1)
        else:
            raise ValueError("Common noise type not determined")

    def train(self):
        self.net.train()

        for i in range(self.n_epochs):

            dW = self.generate.brownian_increments(self.batch_size)
            if self.math_model.jump_type == "none":
                dP = torch.zeros(self.batch_size, self.math_model.N, self.math_model.dim_n, 1)
            else:
                dP = self.generate.compound_poisson_increments(self.batch_size)
            common_noise = self.common_noise(dW, dP)

            _, _, _, loss = self.net(dW, dP, common_noise)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.loss_list.append(float(loss))
            if i % 50 == 0:
                print("Epoch: " + str(i) + ", Loss: " + str(float(loss)))

    def eval(self):
        self.net.eval()

        dW = self.generate.brownian_increments(self.eval_size)
        if self.math_model.jump_type == "none":
            dP = torch.zeros(self.eval_size, self.math_model.N, self.math_model.dim_n, 1)
        else:
            dP = self.generate.compound_poisson_increments(self.eval_size)
        common_noise = self.common_noise(dW, dP)

        state_seq, control_seq, cex_seq, loss = self.net(dW, dP, common_noise)

        return state_seq, control_seq, cex_seq, loss, dW, dP, common_noise


    def plot_loss(self):
        plt.plot(self.loss_list)
        plt.show()

    def plot_realisations(self, state_seq, control_seq, cex_seq, state_seq_true, control_seq_true, cex_seq_true):
        t = torch.linspace(0, self.math_model.T, self.math_model.N)

        i = np.random.randint(state_seq.shape[0])

        plt.plot(t, control_seq[i, :, 0].detach().numpy(), color="red", label="Predicted", linewidth=2.0)
        plt.plot(t, control_seq_true[i, :, 0].detach().numpy(), color="blue", label="Analytical", linewidth=2.0)
        # plt.plot(t,opt_c_true2[i,:,0].detach().numpy(),color="black", label="Analytical2", linewidth=2.0)
        plt.legend(prop={'size': 18})
        plt.xlabel("Time", fontsize=18)
        plt.ylabel("Control", fontsize=18)
        plt.savefig(self.path + "control")
        plt.show()

        plt.plot(t, cex_seq[i, :, 0].detach().numpy(), color="red", label="Predicted", linewidth=2.0)
        plt.plot(t, cex_seq_true[i, :, 0].detach().numpy(), color="blue", label="Analytical", linewidth=2.0)
        # plt.plot(t,x_true[i,:,0].detach().numpy(),color="green",label="Analytical", linewidth=2.0)
        plt.legend(prop={'size': 18})
        plt.xlabel("Time", fontsize=18)
        plt.ylabel("Cond. Expectation", fontsize=18)
        plt.savefig(self.path + "cex")
        plt.show()

        plt.plot(t, state_seq[i, :, 0].detach().numpy(), color="red", label="Predicted", linewidth=2.0)
        plt.plot(t, state_seq_true[i, :, 0].detach().numpy(), color="blue", label="Analytical", linewidth=2.0)
        plt.legend(prop={'size': 18})
        plt.xlabel("Time", fontsize=18)
        plt.ylabel("State", fontsize=18)
        plt.savefig(self.path + "x")
        plt.show()




