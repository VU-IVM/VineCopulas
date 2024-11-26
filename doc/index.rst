Welcome to VineCopulas's documentation!
===================================

`VineCopulas` is a Python package that is able to:

* Fit both `bivariate <https://vinecopulas.readthedocs.io/en/latest/bivariatecopulas.html#Fitting-a-Copula>`_ and `vine copulas <https://vinecopulas.readthedocs.io/en/latest/vinecopulas.html#Fitting-a-Vine-Copula>`_
* Simulate from both `bivariate <https://vinecopulas.readthedocs.io/en/latest/bivariatecopulas.html#Generating-Random-Samples-from-a-Copula>`_ and `vine copulas <https://vinecopulas.readthedocs.io/en/latest/vinecopulas.html#Generating-Random-Samples-from-a-Vine-Copula>`_
* Allow for both `discrete <https://vinecopulas.readthedocs.io/en/latest/bivariatecopulas.html#Discrete-Variables>`_ as well as  `continuous <https://vinecopulas.readthedocs.io/en/latest/bivariatecopulas.html#Probability-Integral-Transform-with-Fitted-CDF>`_ input data
* Draw conditional samples for any variables of interest with the use of `bivariate copulas <https://vinecopulas.readthedocs.io/en/latest/bivariatecopulas.html#Generating-Conditional-Random-Samples-from-a-Copula>`_ and different `vine <https://vinecopulas.readthedocs.io/en/latest/vinecopulas.html#Conditional-Sampling>`_ structures

Installation
-------

.. code-block:: python

    pip install vinecopulas




Documentation Contents
--------

.. toctree::
   :maxdepth: 1

   Home <self>
   bivariatecopulas
   vinecopulas


Getting Started
---------------

Get started by testing checking out the package functionality using the
`Abalone <http://archive.ics.uci.edu/ml/datasets/Abalone>`__ example
data.

.. code:: python

   import pandas as pd
   import matplotlib.pyplot as plt
   import numpy as np
   from vinecopulas.marginals import *
   from vinecopulas.bivariate import *
   from vinecopulas.vinecopula import *

   datapath = 'https://raw.githubusercontent.com/VU-IVM/vinecopula/develop/doc/sample_data.csv'
   df = pd.read_csv(datapath)
   df.head()

.. image:: https://raw.githubusercontent.com/VU-IVM/VineCopulas/main/doc/table_head.JPG
   :width: 500px
   :alt: dataset table

Transform the data to pseudo data and fit a survival gumbel copula
between two variables. Use the fitted copula to generate random samples.

.. code:: python

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

.. image:: https://raw.githubusercontent.com/VU-IVM/VineCopulas/main/doc/bivariate_example.png
   :width: 600px
   :alt: bivariate example

Fit a vine copula between multiple variables in the data, considering
all possible copulas available in the package.

.. code:: python

   cops = list(range(1,16)) # fit vine copula according to these copulas
   M, P, C = fit_vinecop(u, cops, vine = 'R') # fit R-vine
   plotvine(M,variables = list(df.columns[:-1]), plottitle = 'R-Vine') # plot structure

.. image:: https://raw.githubusercontent.com/VU-IVM/VineCopulas/main/doc/vine_structure.JPG
   :width: 600px
   :alt: vine structure
.. image:: https://raw.githubusercontent.com/VU-IVM/VineCopulas/main/doc/vine_example.png
   :width: 600px
   :alt: vine example

For more examples, please have a look at the rest of the 
`documentation <https://vinecopulas.readthedocs.io/en/latest/>`__. 

Contribution Guidelines
-----------------------

Please look at our `Contributing
Guidelines <https://github.com/VU-IVM/VineCopulas/blob/02c24201411677f6968e0f0caefc749c74796715/CONTRIBUTING.md>`__
if you are interested in contributing to this package.

Asking Questions and Reporting Issues
-------------------------------------

If you encounter any bugs or issues while using ``VineCopulas``, please
report them by opening an *issue* in the GitHub repository. Be sure to
provide detailed information about the problem, such as steps to
reproduce it, including operating system and Python version.

If you have any suggestions for improvements, or questions, please also
raise an *issue*.
