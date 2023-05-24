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
# %%
def K_sub(i, j, X1, X2 = None):
    
    ### X1, X2 of shape N/M x d
    if X2 is None:
        tau = (X1.unsqueeze(1) - X1).cpu() ### N x N x d
    else:
        tau = (X1.unsqueeze(1) - X2).cpu() ### N x M x d

    if i == j: ### simple covariance for one channel

        ### alpha_ij
        variance = var()[i].cpu()
        alpha = weight()[i].cpu() ** 2 * torch.float_power(2.0*torch.pi, torch.tensor([tau.shape[-1]/2.0])) * variance.prod(dim=1).sqrt().cpu()
        ### exp(arg_ij)
        exp_arg = -0.5 * torch.einsum("nmd,qd->qnm", tau**2, variance)
        exp = torch.exp(exp_arg)
        ### cos(arg_ij)
        cos_arg = 2.0*numpy.pi * torch.einsum("nmd,qd->qnm", tau, mean()[i].cpu())
        cos = torch.cos(cos_arg)

        Kq = alpha[:,None,None] * exp * cos  # QxNxM

    return torch.sum(Kq, dim=0)


# %%
