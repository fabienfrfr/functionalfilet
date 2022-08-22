# fabienfrfr 20220819
"""
	Universal Approximation Theorem

This very simple case makes it possible to understand the use of the model in the case 
of a regression.

In this example, the model is overtrained for a single function, but highlights problems 
of non-monotonic logical functions (or non-linearly separable, non-connected).

https://machinelearningmastery.com/types-of-classification-in-machine-learning/

"""

## package
import torch, torchvision
import os

## model import
from functionalfilet import model as ff 

## dataset label y / feature x
data_path = os.path.expanduser('~')+'/Dataset/MNIST'
Transforms = torchvision.transforms.Compose([torchvision.transforms.Resize((14,14)), torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.1307,), (0.3081,))])
mnist_dataset = torchvision.datasets.MNIST(data_path, download=True, transform=Transforms)

# extract dataset with transform (tips)
data_loader = torch.utils.data.DataLoader(mnist_dataset, batch_size= mnist_dataset.train_labels.shape[0], shuffle=True)
for batch_idx, (data, target) in enumerate(data_loader) : None

# [N, 1, H, L] -> [N, H, L]
data = data.squeeze()

## ff model
model = ff.FunctionalFilet()

## fit
model.fit(data,target)

## predict
y_pred = model.predict(x)

# show
import pylab as plt
plt.plot(x,y,x,y_pred)
plt.show()