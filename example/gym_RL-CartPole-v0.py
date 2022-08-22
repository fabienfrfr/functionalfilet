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

## model import
from functionalfilet import model as ff 

## function to approximate
f = lambda x : torch.cos(2*x) + x*torch.sin(3*x) + x**0.5

## ff model
model = ff.FunctionalFilet(TYPE="regress")

## to fit
x = torch.linspace(0,10,100)[None]
y = f(x)

## fit
model.fit((x,y))

## predict
y_pred = model.predict(x)

# show
import pylab as plt
plt.plot(x,y,x,y_pred)
plt.show()