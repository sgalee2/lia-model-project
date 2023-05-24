from data import loader_
import torch as t
import matplotlib.pyplot as plt

vox_dataset = loader_.Vox256Embedding("../../data/vox_a_matrices/train")

import numpy as np
import mogptk


trials = 10
index_ = np.random.randint(low=0, high=vox_dataset.__len__(), size=trials)

models = []

for index in index_:
    print(index)
    #get a_matrix & make data right format for mogptk package
    a_ = vox_dataset.__getitem__(index)
    f, d = a_.shape 
    f_span = t.arange(1, f+1).numpy()
    names = ["dim" + str(i+1) for i in range(20)]
    y_s = []
    
    for j in range(d):
        y_s.append(a_[:,j].numpy())
    
    #initialise model, transform data to be 0 mean, unit variance
    dataset = mogptk.DataSet(f_span, y_s, names=names)
    dataset.transform(mogptk.TransformStandard())
    
    #drop 20% of frames picked uniformly
    for channel in dataset:
        channel.remove_randomly(pct=0.2)
    
    # drop relative ranges, 10%
    for i in range(d):
      range_high = np.random.uniform(low=0.1, high=1.0)
      range_low = range_high - 0.1
      dataset[i].remove_relative_range(range_low, range_high)

    model = mogptk.MOSM(dataset, Q=d)
    model.init_parameters(method='LS')
    models.append(model)


for model in models:
    model.train(lr=0.1, verbose=True, iters=500)