# %%
import sys
sys.path.append("../../../lia-model-project")
# %%
import torch, gpytorch
from torch.nn import Parameter
from gpytorch.module import Module
from typing import Optional, Tuple, Union
from latent_gp.gpytorch_model.settings import settings
class MOSM_GP(Module):

    def __init__(self, kernel_module, ):
        super(MOSM_GP, self).__init__()

        self.kernel = kernel_module
        gaussian_scale = weight = Parameter( torch.ones(*kernel_module.batch_shape, kernel_module.output_dims,device=settings.device) )
        self.register_parameter(name="raw_gaussian_noise", parameter=gaussian_scale)
        self.register_constraint("raw_gaussian_noise", gpytorch.constraints.GreaterThan(0.05))

    @property
    def gaussian_noise(self):
        return self.raw_gaussian_noise_constraint.transform(self.raw_gaussian_noise)

    @gaussian_noise.setter
    def gaussian_noise(self, value: Union[torch.Tensor, float]):
        self._set_gaussian_noise(value)

    def _set_gaussian_noise(self, value: Union[torch.Tensor, float]):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_gaussian_noise)
        self.initialize(raw_gaussian_noise=self.raw_gaussian_noise_constraint.inverse_transform(value))

    def __call__(self, x_s, x2_s = None):
        res_dims = []
        for i in range(self.kernel.output_dims):
            res_dims.append(x_s[i].shape[1])
        res = torch.empty(sum(res_dims), sum(res_dims)).to(settings.device)

        m = self.kernel.output_dims        
        cum_dims = [sum(res_dims[0:i]) for i in range(m+1)]
        for i in range(0,m):
            for j in range(i,m):
                if i==j:
                    res[cum_dims[i]:cum_dims[i+1], cum_dims[j]:cum_dims[j+1]] = self.kernel.Ksub(i, j, x_s[i], x_s[j]) + self.gaussian_noise[0,i] * torch.eye(cum_dims[i+1] - cum_dims[i]).to(settings.device)
                else:
                    ksub = self.kernel.Ksub(i, j, x_s[i], x_s[j])[0,:,:]
                    res[cum_dims[i]:cum_dims[i+1], cum_dims[j]:cum_dims[j+1]] = ksub
                    res[cum_dims[j]:cum_dims[j+1], cum_dims[i]:cum_dims[i+1]] = ksub.T
        return res
# %%
