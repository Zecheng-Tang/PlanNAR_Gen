import torch
import torch.nn.functional as F


class Max_Discriminator(torch.nn.Module):
    """
    Discriminator for calculating mutual information maximization
    """
    def __init__(self, hidden_g, initrange):
        super().__init__()
        self.l0 = torch.nn.Linear(2*hidden_g, 1) 
        self.init_weights(initrange)


    def init_weights(self, initrange):
        self.l0.weight.data.uniform_(-initrange, initrange)
        self.l0.bias.data.zero_()

    def forward(self, f_g, f_d):
        h = torch.cat((f_g, f_d), dim=1)
        h = self.l0(F.relu(h))

        return h