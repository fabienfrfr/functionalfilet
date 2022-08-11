**Create your own python package :**

	- pip install setuptools
	- python setup.py sdist bdist_wheel
	- pip install twine
	- twine upload dist/*

**Versionning your project :**

	- git branch v0.0.1 				# create branch
	- git checkout v0.0.1 				# go to branch
	- git push -u origin v0.0.1 		# after add & commit
	- git checkout master && git merge v0.0.1 	# fusion
	- Don't do : "git branch -d v0.0.1 "		# delete

**Testing your package in virtual environment :**

	- python -m venv $HOME/venv
	- source $HOME/venv/bin/activate
	- pip install functionalfilet
		- pip intall ipython
	- python -m IPython
	- # testing
	- deactivate
	- rm -r $HOME/venv

