# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 11:25:45 2023

@author: adayr
"""

import torch
import gpflow
import numpy as np
from math import *
import matplotlib.pyplot as plt
        


f, d_output = 150, 20

d_input = 1  # number of input dimensions
L = 2  # number of latent GPs

def make_sin_taylor(x, term):

    y = torch.zeros_like(x, dtype=float)
    
    for k in range(0,term,1):
        y += ((-1)**k)*(x**(1+2*k))/factorial(1+2*k)
    
    return y
    
train_x = torch.arange(1,f+1)
train_y = torch.zeros([f,d_output])
y_span = torch.linspace(0,5,d_output)

plt.figure(figsize=[12,8])

for i in range(f):
    
    term = torch.randint(low=1,high=11,size=[1]).item()
    train_y[i] = make_sin_taylor(y_span, term) + 0.2 * torch.randn(size=[d_output])
    plt.plot(y_span, train_y[i])
    
