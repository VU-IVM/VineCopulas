# VineCopulas
<img align="right" width="200" alt="Logo" src="https://github.com/VU-IVM/VineCopulas/blob/b67ac132b4d48e316f819b8a467d40e6d1bff146/doc/logogif.gif">


[![github repo badge](https://img.shields.io/badge/github-repo-000.svg?logo=github&labelColor=gray&color=blue)](https://github.com/VU-IVM/VineCopulas)
[![github license badge](https://img.shields.io/github/license/VU-IVM/VineCopulas)](https://github.com/VU-IVM/VineCopulas)
[![Documentation Status](https://readthedocs.org/projects/vinecopulas/badge/?version=latest)](https://vinecopulas.readthedocs.io/en/latest/?badge=latest) 
[![PyPI version](https://badge.fury.io/py/VineCopulas.svg)](https://badge.fury.io/py/VineCopulas)

A pure python implementation for vine copulas

## Installation

```
pip install vinecopulas
```

## Getting Started 

Get started by testing checking out the package functionality using the [Abalone](http://archive.ics.uci.edu/ml/datasets/Abalone) example data.

```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from vinecopulas.marginals import *
from vinecopulas.bivariate import *

datapath = 'https://raw.githubusercontent.com/VU-IVM/vinecopula/develop/doc/sample_data.csv'
df = pd.read_csv(datapath)
df.head()
```
<img src="doc/table_head.JPG" width="300">

Transform the data to pseudo data and fit a survival gumbel copula between two variables. Use the fitted copula to generate random samples.

```python
x = np.array(df)[:,:-1]
u =  pseudodata(x) # computes the pseudodata
cop = 4 # copula 4 is the gumbel copula with 180 degree rotation
par = fit(cop, u[:,:2]) # fit the variables in the first 2 columns 
n = len(u) # number of samples to generate 
ur = random(cop, par, n) # generate random samples
# plot
plt.scatter(u[:,0],u[:,1], label = 'Data')
plt.scatter(ur[:,0], ur[:,1], alpha = 0.5, label = 'Random')
plt.xlabel('$u_1$')
plt.ylabel('$u_2$')
plt.legend()
```
<img src="doc/bivariate_example.png" width="300">

Fit a vine copula between multiple variables in the data, considering all possible copulas available in the package.

```python
cops = list(range(1,16)) # fit vine copula according to these copulas
M, P, C = vinecop(u, cops, vine = 'R') #fit R-vine
plotvine(M,variables = list(df.columns[:-1]), plottitle = 'R-Vine') #plot structure
```

## Contribution Guidelines
---

**Please note:** This package is still in development phase. In case of any problems, or if you have any suggestions for improvements, please raise an *issue*. 
