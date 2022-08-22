# fabienfrfr 20220819

"""
	Universal Approximation Theorem

This very simple case makes it possible to understand the use of the model in the case 
of a regression.

In this example, the model is overtrained for a single function, but highlights non-monotonic 
logical functions problems (or non-linearly separable, non-connected).

"""

## package
import torch
import numpy as np, pylab as plt

## model import
from functionalfilet import model as ff 

## function to approximate
f = lambda x : torch.cos(2*x) + x*torch.sin(3*x) + x**0.5

## ff model
model = ff.FunctionalFilet(train_size=1e4, TYPE="regress", INVERT="same")#, multiprocessing=True)

## feature/feature (f:R -> R)
N = 2
X = torch.cat([torch.linspace(i,10+i,100)[None] for i in range(N)])
y = f(X)

## fit
model.fit(X,y)

## evolution of predict
for i,g in model.test.groupby('IDX_SEED') :
	evo = np.concatenate([p[None] for p in g.PRED])
	# first pred batch (1st gen to last gen)
	for e in evo :
		plt.plot(e[0])
	plt.show(); plt.close()

## predict for all seeder
x, y_ = X.detach().numpy().squeeze(), y.detach().numpy().squeeze()
for i in range(model.NB_SEEDER):
	y_pred = model.predict(X, numpy=True)

	# show curve
	for n in range(N):
		plt.plot(x[n],y_[n],x[n],y_pred[n])
	plt.show(); plt.close()
	# show 'correlation'
	for n in range(N):
		plt.plot(y_[n],y_pred[n])
	plt.show(); plt.close()
	# show graph
	if not(model.SEEDER_LIST[i].control) :
		model.SEEDER_LIST[i].graph.SHOW_GRAPH(LINK_LAYERS = False)
		plt.close()