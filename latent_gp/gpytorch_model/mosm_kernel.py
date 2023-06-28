# %%
import sys
sys.path.append("../../../lia-model-project")
#sys.path.append('C://Users/adayr/OneDrive/Documents/lia_model_project/lia-model-project')

from data.loader_ import *
# %%
import torch, gpytorch
import numpy as np
from typing import Optional, Tuple, Union
# %%
from torch.nn import Parameter
from gpytorch.module import Module
from settings import settings

class MOSM_Kernel(gpytorch.kernels.Kernel):

    """
    Args:
            -- num_components
            -- input_dims
            -- output dims
            -- ard_num_dims: Optional
            -- batch_shape: Optional
    """
    
    def __init__(self,
                num_components,
                input_dims,
                output_dims,
                ard_num_dims: Optional[int] = 1,
                batch_shape:  Optional[torch.Size] = torch.Size([]),
                **kwargs):
        
        super(MOSM_Kernel, self).__init__(ard_num_dims=ard_num_dims, batch_shape=batch_shape)

        self.Q, self.batch_shape, self.input_dims, self.output_dims = num_components, batch_shape, input_dims, output_dims

        ### b_n x m x Q
        weight = Parameter( torch.rand(*self.batch_shape, self.output_dims, self.Q,device=settings.device) )
        phase = Parameter( torch.rand(*self.batch_shape, self.output_dims, self.Q,device=settings.device) )
        self.register_parameter(name="raw_mixture_weights", parameter=weight)
        self.register_parameter(name="phase", parameter=phase)

        ### b_n x m x Q x d
        mean = Parameter( torch.rand(*self.batch_shape, self.output_dims, self.Q, self.input_dims,device=settings.device) )
        variance = Parameter( torch.rand(*self.batch_shape, self.output_dims, self.Q, self.input_dims,device=settings.device) )
        delay = Parameter( torch.rand(*self.batch_shape, self.output_dims, self.Q, self.input_dims,device=settings.device) )
        self.register_parameter(name="raw_mixture_means", parameter=mean)
        self.register_parameter(name="raw_mixture_variance", parameter=variance)
        self.register_parameter(name="delay", parameter=delay)

        ### parameter constraints
        mixture_constraint = gpytorch.constraints.Positive()
        self.register_constraint("raw_mixture_variance", mixture_constraint)
        self.register_constraint("raw_mixture_means", mixture_constraint)
        self.register_constraint("raw_mixture_weights", mixture_constraint)

        self.twopi = torch.float_power(2.0*torch.pi, torch.tensor([self.input_dims/2.0])).to(settings.device)

    @property
    def mixture_variance(self):
        return self.raw_mixture_variance_constraint.transform(self.raw_mixture_variance)

    @mixture_variance.setter
    def mixture_variance(self, value: Union[torch.Tensor, float]):
        self._set_mixture_variance(value)

    def _set_mixture_variance(self, value: Union[torch.Tensor, float]):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_mixture_variance)
        self.initialize(raw_mixture_variance=self.raw_mixture_variance_constraint.inverse_transform(value))

    @property
    def mixture_means(self):
        return self.raw_mixture_means_constraint.transform(self.raw_mixture_means)

    @mixture_means.setter
    def mixture_means(self, value: Union[torch.Tensor, float]):
        self._set_mixture_means(value)

    def _set_mixture_means(self, value: Union[torch.Tensor, float]):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_mixture_means)
        self.initialize(raw_mixture_means=self.raw_mixture_means_constraint.inverse_transform(value))

    @property
    def mixture_weights(self):
        return self.raw_mixture_weights_constraint.transform(self.raw_mixture_weights)

    @mixture_weights.setter
    def mixture_weights(self, value: Union[torch.Tensor, float]):
        self._set_mixture_weights(value)

    def _set_mixture_weights(self, value: Union[torch.Tensor, float]):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_mixture_weights)
        self.initialize(raw_mixture_weights=self.raw_mixture_weights_constraint.inverse_transform(value))

        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_delay)
        self.initialize(raw_delay=self.raw_delay_constraint.inverse_transform(value))
    
    def Ksub(self, i, j, X1, X2=None):
        """
        i,j   -- int:          channels of X1 and X2
        X1,X2 -- torch.Tensor: inputs, of size (b_n, N/M, d)

        ** NEEDS TO BE VECTORISED **
        """
        
        if X2 is None:
            tau = (X1.unsqueeze(-2) - X1.unsqueeze(-3)) ### b_n x N x N x d
        else:
            tau = (X1.unsqueeze(-2) - X2.unsqueeze(-3)) ### b_n x N x M x d

        if i == j: ### covariance across one channel
            
            variance = self.mixture_variance[:,i,:,:] ### b_n x Q x d
            mean = self.mixture_means[:,i,:,:] ### b_n x Q x d
            weight = self.mixture_weights[:,i,:] ### b_n x Q

            alpha = weight ** 2 * self.twopi * variance.prod(dim=-1).sqrt() ### b_n x Q

            exp_arg = -0.5 * torch.einsum("bnmd,bqd -> bqnm", tau ** 2, variance) ### b_n x Q x N x M
            exp = torch.exp(exp_arg) ### b_n x Q x N x M

            cos_arg = 2.0*torch.pi * torch.einsum("bnmd,bqd->bqnm", tau, mean) ### b_n x Q x N x M
            cos = torch.cos(cos_arg) ### b_n x Q x N x M

            K_q = alpha[:, :, None, None] * exp * cos ### b_n x Q x N x M

        else: ### covariance across two different channels
            var_i, var_j = [self.mixture_variance[:,f,:,:] for f in [i,j]] ### both b_n x Q x d
            mean_i, mean_j = [self.mixture_means[:,f,:,:] for f in [i,j]] ### both b_n x Q x d
            w_i, w_j = [self.mixture_weights[:,f,:] for f in [i,j]] ### both b_n x Q
            theta_i, theta_j = [self.delay[:,f,:,:] for f in [i,j]] ### both b_n x Q
            phi_i, phi_j = [self.phase[:,f,:] for f in [i,j]] ### both b_n x Q x d

            inv_vars = 1.0/(var_i + var_j) ### b_n x Q x d
            diff_mean = mean_i - mean_j ### b_n x Q x d
            
            exp_w_arg = -torch.pi ** 2 * torch.sum(diff_mean*inv_vars*diff_mean, dim=-1) ### b_n x Q1
            w_ij = w_i*w_j*torch.exp(exp_w_arg) ### b_n x Q

            mean_ij = inv_vars * (var_i*mean_j + var_j*mean_i) ### b_n x Q x d
            
            var_ij = 2.0 * var_i * inv_vars * var_j ### b_n x Q x d

            theta_ij = theta_i - theta_j ### b_n x Q
            phi_ij = phi_i - phi_j ### b_n x Q x d

            alpha_ij = w_ij * self.twopi * var_ij.prod(dim=-1).sqrt() ### b_n x Q
            tau_delay = tau[:,None,:,:,:]  - theta_ij[:,:,None,None,:] ### b_n x Q x N x M x d

            exp_arg = -0.5 * torch.einsum("bqnmd,bqd -> bqnm", tau_delay ** 2, var_ij) ### b_n x Q x N x M
            exp = torch.exp(exp_arg) ### b_n x Q x N x M

            cos_arg = 2.0*torch.pi * torch.einsum("bqnmd,bqd->bqnm", tau_delay, mean_ij) + phi_ij[:,:,None,None]
            cos = torch.cos(cos_arg) ### b_n x Q x N x M


            K_q = alpha_ij[:, :, None, None] * exp * cos ### b_n x Q x N x M

        return torch.sum(K_q, dim=1) ### b_n x N x M
    
#%%
if __name__ == "__main__":
    vox_dataset = Vox256Embedding(r"C:\Users\adayr\OneDrive\Documents\lia_model_project\lia-model-project\data\vox_a_matrices\train")
    index = np.random.randint(low=0, high=vox_dataset.__len__())
    a_ = vox_dataset.__getitem__(index)
    f, d = a_.shape 
    f_span = t.arange(1, f+1,dtype=torch.float64)

    x = f_span[None,:,None].to(settings.device)
    print(x.shape[1])
    y_s, y_means, y_stds = [], [], []
    
    for j in range(d):
        y = a_[:,j].to(settings.device)
        y_mean, y_std = y.mean(), y.std()
        y_means.append(y_mean)
        y_stds.append(y_std)
        y_s.append((y-y_mean)/y_std)
    kern = MOSM_Kernel(5, 1, 20, batch_shape=torch.Size([1]), ard_num_dims=1)
    from mosm_model import MOSM_GP
    gp = MOSM_GP(kern)
# %%
if __name__ == "__main__":
    x_s = []
    for i in range(20):
        x_s.append(x[0:1,:,:])
    Y = torch.hstack(y_s).reshape(-1,1)
# %%
if __name__ == "__main__":
    def nll(gram, target):
        L = torch.linalg.cholesky(gram)
        ldet = 2*sum(torch.log(L.diag()))

        res = target.T @ torch.cholesky_solve(target, L)
        return res + ldet
    optim = torch.optim.Adam(gp.parameters(),lr=0.1)
    for i in range(100):
        
        optim.zero_grad()
        k = gp(x_s)
        loss = nll(k, Y)
        loss.backward()
        print(i,loss)
        optim.step()
# %%
