<h1 align="center">
<img src="/branding/logo.svg" width="500">
</h1><br>

###### Attribution required : Fabien Furfaro (CC 4.0 BY NC ND SA)

(Work in developpement)

See the notebook [here](/.ipynb) (Work in progress)

An Artificial Neural Network Functionalized by Evolution - This model is based on this [paper](https://arxiv.org/abs/2205.10118).


**Requirements :**

	numpy
	pandas
	torch
	torchvision
	networkx


**Create your own python package :**

	- pip install setuptools
	- python setup.py sdist bdist_wheel
	- pip install twine
	- twine upload dist/*

**Versionning your project :**

	- git branch v0.0.1 				# create branch
	- git checkout v0.0.1 				# go to branch
	- git checkout master && git merge v0.0.1 	# fusion
	- Don't do : "git branch -d v0.0.1 "		# delete