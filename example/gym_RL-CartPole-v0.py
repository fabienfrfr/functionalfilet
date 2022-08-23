# fabienfrfr 20220819
"""
	Reinforcement Q-learning

This very simple case makes it possible to understand the use of the model in the case 
of a markov decision process.

In this example, the model is adapted for unsupervised learning. It's highlights use
decomposition model (step per step) and dilemna problematic's in evolution case.

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