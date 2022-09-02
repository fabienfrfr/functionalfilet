# fabienfrfr 20220819

"""
	Universal Approximation Theorem

This very simple case makes it possible to understand the use of the model in the case 
of a regression.

In this example, the model is overtrained for a single function, but highlights non-monotonic 
logical functions problems (or non-linearly separable, non-connected).

"""

## package
import torch, os, ast
import numpy as np, pylab as plt

## model import
from functionalfilet import model as ff 

## function to approximate
f = lambda x : torch.cos(2*x) + x*torch.sin(3*x) + x**0.5

### feature/feature (f:R -> R)
N = 2
# Local (linspace) & unlocal (random) correlation
torch.manual_seed(0) #reproductibility
X = torch.cat([torch.linspace(i,10+i,100)[None] for i in range(N)]+[10*torch.rand(N,100)])
y = f(X)

## ff model
model = ff.FunctionalFilet(train_size=1e4, TYPE="regress", INVERT="same")#, multiprocessing=True)
load_name = 'regress_2220823_134320'

## fit (or load)
path = os.path.expanduser('~')+'/Saved_Model/ff_' + load_name
if os.path.isdir(path):
	### LOAD
	model.load(path)
else :
	### FIT
	model.fit(X,y)

## evolution of predict
for i,g in model.test.groupby('IDX_SEED') :
	print("[INFO] Predict evolution training of seeder : " + str(i))
	fig, ax = plt.subplots()
	evo = np.concatenate([np.array(ast.literal_eval(p))[None] for p in g.PRED])
	# first pred batch (1st gen to last gen)
	for e in evo :
		ax.plot(e[0])
	plt.show(); plt.close()

## predict for all seeder
x, y_ = X.detach().numpy().squeeze(), y.detach().numpy().squeeze()
for i in range(len(model.SEEDER_LIST)):
	y_pred = model.predict(X, index=i, numpy=True)

	# show curve
	fig, ax = plt.subplots()
	print("[INFO] Predict curve of seeder : " + str(i))
	for n in range(N):
		ax.plot(x[n],y_[n],x[n],y_pred[n])
	plt.show(); plt.close()
	# show 'correlation'
	fig, ax = plt.subplots()
	print("[INFO] Correlation curve of seeder : " + str(i))
	for n in range(N):
		ax.plot(y_[n],y_pred[n])
	plt.show(); plt.close()
	# show graph
	if not(model.SEEDER_LIST[i].control) :
		print("[INFO] Neural network graph of seeder : " + str(i))
		model.SEEDER_LIST[i].graph.SHOW_GRAPH(LINK_LAYERS = False)
		plt.close()