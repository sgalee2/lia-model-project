# %%
import torch

class Settings:

    dtype=torch.float64

    if torch.cuda.is_available:
        device=torch.device('cuda', torch.cuda.current_device())
    else:
        device=torch.device('cpu')

settings = Settings()

def use_dtype(dtype=torch.float64):
    settings.dtype=dtype
# %%
