# fabienfrfr 20220819
"""
	Blob classification with unbalanced data & cluster

This very simple case makes it possible to understand the use of the model in the case 
of a classification.

In this example, the model is overtrained, but highlights problems of overlapping
and unbalanced data.

"""

## package
import os, numpy as np
import torch, pandas as pd
from sklearn.datasets import make_classification

import pylab as plt, seaborn as sns

## model import
from functionalfilet import model as ff 

# define dataset
X, y = make_classification(n_samples=1000, n_features=3, n_redundant=0, n_informative=3, n_clusters_per_class=2, n_classes=4, weights=[0.5,0.3,0.15,0.05], random_state=1)

# Fast EDA
df = pd.DataFrame(np.concatenate((y[:,None],X), axis=1), columns=['label','posX','posY', 'posZ'])
print("[INFO] EDA : Pairplot of small dataset.")
sns.pairplot(df, hue="label"); plt.show()
plt.close()

# to torch
X, y = torch.tensor(X, dtype=torch.float), torch.tensor(y).type(torch.LongTensor)

## ff model
model = ff.FunctionalFilet(train_size=1e5, INVERT=True)
load_name = 'class_20220823_180834'

## fit (or load)
path = os.path.expanduser('~')+'/Saved_Model/ff_' + load_name
if os.path.isdir(path):
	### LOAD
	model.load(path)
else :
	### FIT
	model.fit(X,y)

# predict
for i in range(len(model.SEEDER_LIST)):
	y_pred = model.predict(X, index=i)
	# plot the dataset and color the by class label
	y_max = torch.argmax(y_pred, dim=1).cpu().numpy()
	for label in np.unique(y_max):
		row_ix = np.where(y_max == label)[0]
		plt.scatter(X[row_ix, 0], X[row_ix, 1], label=str(label))
	plt.show();plt.close()
