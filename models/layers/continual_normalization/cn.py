from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn import functional as F
import torch


class _CN(_BatchNorm):
    def __init__(self, target, layer_number, eps=1e-5, momentum=1, affine=True):
        num_features = target.num_features
        self.training = target.training
        super(_CN, self).__init__(num_features, eps, momentum, affine=True)
        self.running_mean = target.running_mean
        self.running_var = target.running_var
        self.first_run = True
        self.out_gn = None
        self.weight = target.weight
        self.bias = target.bias
        self.N = num_features
        self.setG()
        self.num_epoch = 0
        self.layer_number = layer_number
        self.group_running_mean = []
        self.group_running_var = []
        self.b_size = []

    def setG(self):
        pass

    def load_group_vars(self, group_running_mean, group_running_var, b_size, device):
        self.group_running_mean = torch.Tensor(group_running_mean).to(device)
        self.group_running_var = torch.Tensor(group_running_var).to(device)
        self.b_size = b_size
        self.first_run = False
        return

    def get_group_vars(self):
        return self.group_running_mean, self.group_running_var, self.b_size

    def _check_input_dim(self, input):
        pass

    def forward(self, input):

        bs = input.shape[0]

        g_size = int(input.shape[1] / self.G)
        w, h = input.shape[2], input.shape[2]
        entery = input.clone().detach()

        self.total_mean = entery.reshape((bs, self.G, g_size, w, h)).mean([2, 3, 4]).to(input.device)
        self.total_var = entery.reshape((bs, self.G, g_size, w, h)).contiguous().view([bs, self.G, -1]).var(2,
                                                                                                            unbiased=False).to(
            input.device)

        if self.first_run:
            self.group_running_mean = self.total_mean
            self.group_running_var = self.total_var
            self.first_run = False
            self.b_size = input.shape[0]

        if self.training and not self.first_run and self.b_size == input.shape[0]:
            self.group_running_mean = (1 - self.momentum) * self.group_running_mean + self.momentum * self.total_mean
            self.group_running_var = (1 - self.momentum) * self.group_running_var + self.momentum * self.total_var

        self.out_gn = F.group_norm(input, self.G, None, None, self.eps)
        out = F.batch_norm(self.out_gn, self.running_mean, self.running_var, self.weight, self.bias,
                           self.training, self.momentum, self.eps)
        self.num_epoch += 1
        return out


class CN4(_CN):
    def setG(self):
        self.G = 4


class CN8(_CN):
    def setG(self):
        self.G = 8


class CN16(_CN):
    def setG(self):
        self.G = 16


class CN32(_CN):
    def setG(self):
        self.G = 32


class CN64(_CN):
    def setG(self):
        self.G = 64


class CN40(_CN):
    def setG(self):
        self.G = 40
