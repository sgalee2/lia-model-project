from data import loader_
import torch as t
import matplotlib.pyplot as plt

vox_dataset = loader_.Vox256Embedding("../../data/vox_a_matrices/train")

import numpy as np
import mogptk

np.random.seed(123)

index = np.random.randint(low=0, high=17000)

a_ = vox_dataset.__getitem__(index)
f, d = a_.shape 
f_span = t.arange(1, f+1).numpy()
names = ["dim" + str(i+1) for i in range(20)]
y_s = []

for j in range(d):
    y_s.append(a_[:,j].numpy())
    
dataset = mogptk.DataSet(f_span, y_s, names=names)
dataset.transform(mogptk.TransformStandard())
model = mogptk.MOSM(dataset, Q=d)

for channel in dataset:
    channel.remove_randomly(pct=0.4)

# drop relative ranges to simulate sensor failure
for i in range(d):
  range_high = np.random.uniform(low=0.1, high=1.0)
  range_low = range_high - 0.1
  dataset[i].remove_relative_range(range_low, range_high)
  
dataset.plot()

model.plot_prediction()

model.init_parameters(method='LS')
model.plot_prediction()

model.train(iters=200, lr=0.2, verbose=True, error='MAE', plot=True)

model.plot_prediction()