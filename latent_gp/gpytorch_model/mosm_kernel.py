# %%
import sys
sys.path.append("../../lia-model-project")
# %%
import torch, gpytorch
from typing import Optional, Tuple, Union
# %%
from torch.nn import Parameter

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
        weight = Parameter( torch.ones(*self.batch_shape, self.output_dims, self.Q) )
        phase = Parameter( torch.zeros(*self.batch_shape, self.output_dims, self.Q) )
        self.register_parameter(name="raw_mixture_weights", parameter=weight)
        self.register_parameter(name="raw_phase", parameter=phase)

        ### b_n x m x Q x d
        mean = Parameter( torch.zeros(*self.batch_shape, self.output_dims, self.Q, self.input_dims) )
        variance = Parameter( torch.zeros(*self.batch_shape, self.output_dims, self.Q, self.input_dims) )
        delay = Parameter( torch.zeros(*self.batch_shape, self.output_dims, self.Q, self.input_dims) )
        self.register_parameter(name="raw_mixture_means", parameter=mean)
        self.register_parameter(name="raw_mixture_variance", parameter=variance)
        self.register_parameter(name="raw_delay", parameter=delay)

        ### parameter constraints
        mixture_constraint = gpytorch.constraints.Positive()
        self.register_constraint("raw_mixture_variance", mixture_constraint)
        self.register_constraint("raw_mixture_means", mixture_constraint)
        self.register_constraint("raw_mixture_weights", mixture_constraint)

        from math import inf
        unconstrained = gpytorch.constraints.Interval(-100.,100.)
        self.register_constraint("raw_phase", unconstrained)
        self.register_constraint("raw_delay", unconstrained)

        self.twopi = torch.float_power(2.0*torch.pi, torch.tensor([self.input_dims/2.0]))

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

    @property
    def phase(self):
        return self.raw_phase_constraint.transform(self.raw_phase)

    @phase.setter
    def phase(self, value: Union[torch.Tensor, float]):
        self._set_phase(value)

    def _set_phase(self, value: Union[torch.Tensor, float]):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_phase)
        self.initialize(raw_phase=self.raw_phase_constraint.inverse_transform(value))

    @property
    def delay(self):
        return self.raw_delay_constraint.transform(self.raw_delay)

    @delay.setter
    def delay(self, value: Union[torch.Tensor, float]):
        self._set_delay(value)

    def _set_delay(self, value: Union[torch.Tensor, float]):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_delay)
        self.initialize(raw_delay=self.raw_delay_constraint.inverse_transform(value))
    
    def Ksub(self, i, j, X1, X2=None):
        
        if X2 is None:
            tau = (X1.unsqueeze(-2) - X1.unsqueeze(-3)) ### b_n x N x N x d
        else:
            tau = (X1.unsqueeze(-2) - X2.unsqueeze(-3)) ### b_n x N x M x d

        if i == j: ### covariance across one channel
            variance = self.mixture_variance[:,i,:,:] ### b_n x Q x d
            mean = self.mixture_means[:,i,:,:] ### b_n x Q x d
            weight = self.mixture_weights[:,i,:] ### b_n x Q

            alpha = weight ** 2 * self.twopi * variance.prod(dim=-1).sqrt()

            exp_arg = -0.5 * torch.einsum("bnmd,bqd -> bqnm", tau ** 2, variance)
            exp = torch.exp(exp_arg)

            cos_arg = 2.0*torch.pi * torch.einsum("bnmd,bqd->bqnm", tau, mean)
            cos = torch.cos(cos_arg)

            K_q = alpha[:, :, None, None] * exp * cos

        else: ### covariance across two different channels

            var_i, var_j = self.mixture_variance[:,[i,j],:,:] ### both b_n x Q x d
            mean_i, mean_j = self.mixture_means[:,[i,j],:,:] ### both b_n x Q x d
            w_i, w_j = self.mixture_weights[:,[i,j],:] ### both b_n x Q
            theta_i, theta_j = self.delay[:,[i,j],:,:] ### both b_n x Q
            phi_i, phi_j = self.phase[:,[i,j],:] ### both b_n x Q x d

            inv_vars = 1.0/(var_i + var_j) ### b_n x Q x d
            diff_mean = mean_i - mean_j ### b_n x Q x d
            
            exp_w_arg = -torch.pi ** 2 * torch.sum(diff_mean*inv_vars*diff_mean, dim=-1)
            w_ij = w_i*w_j*torch.exp(exp_w_arg)

            mean_ij = inv_vars * (var_i*mean_j + var_j*mean_i)
            
            var_ij = 2.0 * var_i * inv_vars * var_j

            theta_ij = theta_i - theta_j
            phi_ij = phi_i - phi_j

            alpha_ij = w_ij * self.twopi * var_ij.prod(dim=-1).sqrt()
            tau_delay = tau[:,None,:,:,:]  - theta_ij[:,:,None,None,:] ### b_n x Q x N x M x d

            exp_arg = -0.5 * torch.einsum("bqnmd,bqd -> bqnm", tau_delay ** 2, var_ij) ### b_n x Q x N x M
            exp = torch.exp(exp_arg)

            cos_arg = 2.0*torch.pi * torch.einsum("bqnmd,bqd->bqnm", tau_delay, mean_ij) + phi_ij[:,:,None,None]
            cos = torch.cos(cos_arg) ### b_n x Q x N x M


            K_q = alpha_ij[:, :, None, None] * exp * cos

        return torch.sum(K_q, dim=1)

        

if __name__ == "__main__":
    N, M, b_n, d = 100, 50, 2, 3
    x = torch.rand(2,100,3)
    x2 = torch.rand(2,50,3)
    kern = MOSM_Kernel(5, d, 20, batch_shape=torch.Size([b_n]),ard_num_dims=d)
# %%
