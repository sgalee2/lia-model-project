from data import loader_
import torch as t
import matplotlib.pyplot as plt

vox_dataset = loader_.Vox256Embedding("../data/vox_a_matrices/train")

for i in range(10):
    print(i)
    plt.figure(figsize=[12,8])
    a_ = vox_dataset.__getitem__(i)
    f, d = a_.shape 
    f_span = t.arange(1, f+1)
    
    
    for j in range(d):
        plt.scatter(f_span, a_[:,j])