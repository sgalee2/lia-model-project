# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 11:25:45 2023

@author: adayr
"""

import torch
import gpflow
import mogptk
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
    
train_x = torch.arange(1,f+1).numpy()
train_y = torch.zeros([f,d_output])
y_span = torch.linspace(0,5,d_output)

for i in range(f):
    
    term = torch.randint(low=1,high=11,size=[1]).item()
    train_y[i] = make_sin_taylor(y_span, term) + 0.2 * torch.randn(size=[d_output])
    
names = ["dim" + str(i+1) for i in range(20)]
y_s = []

for j in range(d_output):
    y_s.append(train_y[:,j].numpy())

dataset = mogptk.DataSet(train_x, y_s, names=names)
model = mogptk.MOSM(dataset, Q=d_output)

for channel in dataset:
    channel.remove_randomly(pct=0.4)

# drop relative ranges to simulate sensor failure
for i in range(d_output):
  range_high = np.random.uniform(low=0.1, high=1.0)
  range_low = range_high - 0.1
  dataset[i].remove_relative_range(range_low, range_high)