# 20220810 fabienfrfr
from setuptools import setup, find_packages
import os

here = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(here, "README.rst")) as fh :
    long_description = fh.read()

VERSION = '0.5.1'
DESCRIPTION = 'Evolutionnary Neural Network Model with PyTorch'

# Setting up
setup(
    name="functionalfilet",
    version=VERSION,
    author="FabienFrfr (Fabien Furfaro)",
    author_email="<fabien.furfaro@gmail.com>",
    description=DESCRIPTION,
    long_description=long_description,
    packages=find_packages(),
    install_requires=['numpy', 'pandas', 'torch', 'torchvision', 'matplotlib', 'networkx', 'tqdm','scikit-learn', 'scikit-datasets', 'seaborn', 'gym'],
    keywords=['python', 'pytorch', 'graph', 'machine learning', 'evolution'],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)
