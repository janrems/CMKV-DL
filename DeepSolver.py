import torch
import numpy as np
import matplotlib.pyplot as plt


from Net import Common_Noise
from DataGenerator import DataGenerator



class DeepSolver:
    def __init__(self, math_model, batch_size, eval_size, learning_rate, n_epochs, ridge_param, depth):
        self.math_model = math_model
        self.generate = DataGenerator(math_model)
        self.batch_size = batch_size
        self.eval_size = eval_size
        self.n_epochs = n_epochs
        self.net = Common_Noise(math_model, ridge_param, depth)

        self.optimizer = torch.optim.Adam(self.net.parameters(), learning_rate)
        self.loss_list = []


    def common_noise(self, dW, dP):
        if self.math_model.common_noise == "Brownian":
            return self.generate.process_from_increments(dW[:, :, 0, :])
        elif self.math_model.common_noise == "Poisson":
            return self.generate.process_from_increments(dP[:, :, 0, :])
        elif self.math_model.common_noise == "Mixed":
            Wt = self.generate.process_from_increments(dW[:, :, 0, :])
            Pt = self.generate.process_from_increments(dP[:, :, 0, :])
            return torch.cat((Wt, Pt), dim=-1).unsqueeze(-1)
        else:
            raise ValueError("Common noise type not determined")

    def train(self):
        self.net.train()

        for i in range(self.n_epochs):

            dW = self.generate.brownian_increments(self.batch_size)
            if self.math_model.jump_type == "None":
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
        if self.math_model.jump_type == "None":
            dP = torch.zeros(self.eval_size, self.math_model.N, self.math_model.dim_n, 1)
        else:
            dP = self.generate.compound_poisson_increments(self.eval_size)
        common_noise = self.common_noise(dW, dP)

        state_seq, control_seq, cex_seq, loss = self.net(dW, dP, common_noise)

        return state_seq, control_seq, cex_seq, loss, dW, dP, common_noise


    def plot_loss(self, loss_true):
        epochs = np.linspace(1, len(self.loss_list), len(self.loss_list))
        plt.plot(epochs[:], self.loss_list[:] , label="Estimated value", linewidth=3.0)
        plt.axhline(float(loss_true), color="red", label="True value")
        plt.legend(prop={'size': 14})
        plt.xlabel("Epoch", fontsize=14)
        plt.ylabel("Loss", fontsize=14)
        plt.savefig(self.math_model.path + "loss")
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
        plt.savefig(self.math_model.path + "control")
        plt.show()

        plt.plot(t, cex_seq[i, :, 0].detach().numpy(), color="red", label="Predicted", linewidth=2.0)
        plt.plot(t, cex_seq_true[i, :, 0].detach().numpy(), color="blue", label="Analytical", linewidth=2.0)
        # plt.plot(t,x_true[i,:,0].detach().numpy(),color="green",label="Analytical", linewidth=2.0)
        plt.legend(prop={'size': 18})
        plt.xlabel("Time", fontsize=18)
        plt.ylabel("Cond. Expectation", fontsize=18)
        plt.savefig(self.math_model.path + "cex")
        plt.show()

        plt.plot(t, state_seq[i, :, 0].detach().numpy(), color="red", label="Predicted", linewidth=2.0)
        plt.plot(t, state_seq_true[i, :, 0].detach().numpy(), color="blue", label="Analytical", linewidth=2.0)
        plt.legend(prop={'size': 18})
        plt.xlabel("Time", fontsize=18)
        plt.ylabel("State", fontsize=18)
        plt.savefig(self.math_model.path + "x")
        plt.show()


    def L2(self,state_seq, state_seq_true, control_seq, control_seq_true, cex_seq, cex_seq_true):
        state = torch.mean(self.math_model.dt*torch.sum(torch.linalg.norm(state_seq - state_seq_true,dim=-1) ** 2, dim=1), dim=0) ** 0.5
        state2 = torch.mean(self.math_model.dt*torch.sum(torch.linalg.norm(state_seq_true, dim=-1) ** 2, dim=1), dim=0) ** 0.5
        control = torch.mean(self.math_model.dt*torch.sum(torch.linalg.norm(control_seq - control_seq_true, dim=-1) ** 2, dim=1), dim=0) ** 0.5
        control2 = torch.mean(self.math_model.dt*torch.sum(torch.linalg.norm(control_seq_true,dim=-1) ** 2, dim=1), dim=0) ** 0.5
        cex = torch.mean(self.math_model.dt*torch.sum(torch.linalg.norm(cex_seq - cex_seq_true, dim=-1) ** 2, dim=1), dim=0) ** 0.5
        cex2 = torch.mean(self.math_model.dt*torch.sum(torch.linalg.norm(cex_seq_true, dim=-1) ** 2, dim=1), dim=0) ** 0.5

        return float((state/state2)), float(control/control2), float(cex/cex2)




