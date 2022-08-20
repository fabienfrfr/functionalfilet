# fabienfrfr 20220819
"""
	Universal Approximation Theorem

This very simple case makes it possible to understand the use of the model in the case 
of a regression.

In this example, the model is overtrained for a single function, but highlights problems 
of non-monotonic logical functions (or non-linearly separable, non-connected).

"""

## package
import torch
import pylab as plt

## model import
from functionalfilet import model as ff 

## function to approximate
f = lambda x : torch.cos(2*x) + x*torch.sin(3*x) + x**0.5

## ff model
model = ff.FunctionalFilet(TYPE="regress")

## feature/feature
X = torch.linspace(0,10,100)[None]
y = f(X)

## fit
model.fit(X,y)

## predict for all seeder
x, y_ = X.detach().numpy().squeeze(), y.detach().numpy().squeeze()
for i in range(model.NB_SEEDER):
	y_pred = model.predict(X)

	# show curve
	plt.plot(x,y_,x,y_pred)
	plt.show(); plt.close()
	# show correlation
	plt.plot(y_,y_pred)
	plt.show(); plt.close()