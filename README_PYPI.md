# VineCopulas
<img align="right" width="200" src="https://raw.githubusercontent.com/VU-IVM/VineCopulas/main/doc/logo2.png">


[![github repo badge](https://img.shields.io/badge/github-repo-000.svg?logo=github&labelColor=gray&color=blue)](https://github.com/VU-IVM/VineCopulas)
[![github license badge](https://img.shields.io/github/license/VU-IVM/VineCopulas)](https://github.com/VU-IVM/VineCopulas)
[![Documentation Status](https://readthedocs.org/projects/vinecopulas/badge/?version=latest)](https://vinecopulas.readthedocs.io/en/latest/?badge=latest) 
[![PyPI version](https://badge.fury.io/py/VineCopulas.svg)](https://badge.fury.io/py/VineCopulas)

`VineCopulas` is a Python package that is able to:
* Fit both bivariate and vine copulas
* Simulate from both bivariate and vine copulas
* Allow for both discrete as well as continuous input data
* Draw conditional samples for any variables of interest with the use of bivariate copulas and different vine structures

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
<img src="https://raw.githubusercontent.com/VU-IVM/VineCopulas/main/doc/table_head.JPG" width="500">

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
<img src="https://raw.githubusercontent.com/VU-IVM/VineCopulas/main/doc/bivariate_example.png" width="600">

Fit a vine copula between multiple variables in the data, considering all possible copulas available in the package.

```python
cops = list(range(1,16)) # fit vine copula according to these copulas
M, P, C = fit_vinecop(u, cops, vine = 'R') # fit R-vine
plotvine(M,variables = list(df.columns[:-1]), plottitle = 'R-Vine') # plot structure
```

<img src="https://raw.githubusercontent.com/VU-IVM/VineCopulas/main/doc/vine_structure.JPG" width="600">
<img src="https://raw.githubusercontent.com/VU-IVM/VineCopulas/main/doc/vine_example.png" width="600">

For more examples, please have a look at the [documentation](https://vinecopulas.readthedocs.io/en/latest/).
## Contribution Guidelines
---

Please look at our [Contributing Guidelines](https://github.com/VU-IVM/VineCopulas/blob/02c24201411677f6968e0f0caefc749c74796715/CONTRIBUTING.md) if you are interested in contributing to this package.

## Asking Questions and Reporting Issues

If you encounter any bugs or issues while using `VineCopulas`, please report them by opening an *issue* in the GitHub repository. Be sure to provide detailed information about the problem, such as steps to reproduce it, including operating system and Python version.

If you have any suggestions for improvements, or questions, please also raise an *issue*. 
