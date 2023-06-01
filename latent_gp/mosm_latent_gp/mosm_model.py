#%%
import sys
sys.path.append("../../lia-model-project")

#%%
import torch, gpytorch, mogptk, numpy
import matplotlib.pyplot as plt

#%%
### fake data
N, m, q, d = 100, 20, 5, 1
x = torch.rand(N, d)
y = torch.rand(N, m)
#%%
y_s = []
for i in range(m):
    y_s.append(y[:,i].numpy())
dataset = mogptk.DataSet(x.numpy(), y_s, names=['data'+str(i) for i in range(m)])
#%%
model = mogptk.MOSM(dataset, Q=5)
weight, mean, var, delay, phase, scale = model.get_parameters()
#%%
X1 = model.gpr.X
Y = model.gpr.y
c1 = X1[:,0].long()
m1 = [c1==i for i in range(m)]
x1 = [X1[m1[i],1:] for i in range(m)]
r1 = [torch.nonzero(m1[i], as_tuple=False) for i in range(m)]  # as_tuple avoids warning

#%%
from torch.nn import Module, Parameter

class MOSM_GP(Module):

    """
    - x_s: list [(N_1, d), ..., (N_m, d)]
    - y_s: list [(N_1)   , ..., (N_m)]
    
    """

    def __init__(self, x_s, y_s, Q):

        super(MOSM_GP, self).__init__()

        ### either len(x_s) == len(y_s) <- different inputs for each channel
        ###        len(x_s) == 1        <- same inputs for each channel
        ### then check 
        if len(x_s) == len(y_s) or len(x_s) == 1:
            self.input_dims = x_s[0].shape[1]
            self.output_dims = len(y_s)
            if len(x_s) != 1:
                for m in range(self.output_dims):
                    assert x_s[m].shape[0] == y_s[m].shape[0]
        else:
            raise ValueError("Channels are missing inputs")
        
        self.x_s, self.y_s = x_s, y_s

        self.weight = Parameter( torch.ones(self.output_dims, Q) )
        self.mean = Parameter( torch.zeros(self.output_dims, Q, self.input_dims) )
        self.variance = Parameter( torch.ones(self.output_dims, Q, self.input_dims) )
        self.delay = Parameter( torch.zeros(self.output_dims, Q, self.input_dims) )
        self.phase = Parameter( torch.zeros(self.output_dims, Q) )

        






# %%
def K_sub(i, j, X1, X2 = None):
    
    ### X1, X2 of shape N/M x d
    if X2 is None:
        tau = (X1.unsqueeze(1) - X1) ### N x N x d
    else:
        tau = (X1.unsqueeze(1) - X2) ### N x M x d

    if i == j: ### simple covariance for one channel

        ### alpha_ij
        variance = var()[i]
        alpha = weight()[i] ** 2 * torch.float_power(2.0*torch.pi, torch.tensor([tau.shape[-1]/2.0])) * variance.prod(dim=1).sqrt()
        ### exp(arg_ij)
        exp_arg = -0.5 * torch.einsum("nmd,qd->qnm", tau**2, variance)
        exp = torch.exp(exp_arg)
        ### cos(arg_ij)
        cos_arg = 2.0*numpy.pi * torch.einsum("nmd,qd->qnm", tau, mean()[i])
        cos = torch.cos(cos_arg)

        Kq = alpha[:,None,None] * exp * cos  # QxNxM
    else:
        Kq = torch.zeros_like(tau)

    return torch.sum(Kq, dim=0)


# %%
def K(X1, X2=None):
    ### X1, X2 
    pass