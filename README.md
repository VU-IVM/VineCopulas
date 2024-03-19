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

Gest started by testing checking out the package functionality using the [Abalone](http://archive.ics.uci.edu/ml/datasets/Abalone) example data.

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

## Contribution Guidelines
---

**Please note:** This package is still in development phase. In case of any problems, or if you have any suggestions for improvements, please raise an *issue*. 
