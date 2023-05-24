import sys
sys.path.append('C://Users/adayr/OneDrive/Documents/lia_model_project/lia-model-project')

from data import loader_
import torch as t
import matplotlib.pyplot as plt
import mogptk

vox_dataset = loader_.Vox256Embedding("data/vox_a_matrices/train")

i = t.randint(low=0, high=vox_dataset.__len__(), size=[1])
a_ = vox_dataset.__getitem__(i)
f, d = a_.shape 
f_span = t.arange(1, f+1)
    
f, d = a_.shape 
f_span = t.arange(1, f+1).numpy()
names = ["dim" + str(i+1) for i in range(20)]
y_s = []

for j in range(d):
    y_s.append(a_[:,j].numpy().copy())
    
dataset = mogptk.DataSet(f_span, y_s, names=names)
dataset.transform(mogptk.TransformStandard())
model = mogptk.MOSM(dataset, Q=d)

plot = dataset.plot()