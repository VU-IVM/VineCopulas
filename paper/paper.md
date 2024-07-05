---
title: 'VineCopulas: an open-source Python package for vine copula modelling'
tags:
  - Python
  - copula
  - statistics
authors:
  - name: Judith N. Claassen
    orcid: 0009-0000-2979-1297
    affiliation: 1
  - name: Elco E. Koks
    orcid: 0000-0002-4953-4527
    affiliation: 1
  - name: Marleen C. de Ruiter
    orcid: 0000-0001-5991-8842
    affiliation: 1
  - name: Philip J. Ward
    orcid: 0000-0001-7702-7859
    affiliation: "1, 2"
  - name: Wiebke S. Jäger
    orcid: 0009-0007-8628-6060
    affiliation: 1
affiliations:
 - name: Institute for Environmental Studies, Vrije Universiteit Amsterdam, Amsterdam, The Netherlands
   index: 1
 - name: Deltares, Delft, The Netherlands
   index: 2

date: 19 March 2024
bibliography: paper.bib
---

# Summary
A copula method can be used to describe the dependency structure between several random variables. Copula-methods are used widely in various research fields across different disciplines, ranging from  finance to the bio-geophysical sciences [@dimann_2013_selecting; @klein_2020_bayesian; @Mitskopoulos2022]. While some other multivariate distributions, for instance a multivariate normal distribution, allow for a highly symmetric dependency structure with the same univariate and multivariate marginal distributions, copulas can model the joint distribution of multiple random variables separately from their marginal distribution [@sklar1959fonctions; @Czado2022].

Once a copula distribution has been modelled, they allow for random samples of the data to be generated, as well as conditional samples. For example, if a copula has been fit between people's height and weight, this copula can create random correlated samples of both variables as well as conditional samples, e.g., samples of weight given a specific height.

Although copulas are an excellent tool to model dependencies in bivariate data, data with two variables, there are only a limited number of copulas capable of modelling larger multivariate datasets, for example, the Gaussian and Student-t copula. However, when modelling the dependencies between a large number of different variables, a more flexible multivariate modelling tool may be required that does not assume a single copula to capture all the individual dependencies. To this end, vine copulas have been proposed as a method to construct a multivariate model with the use of bivariate copulas as building blocks [@Joe1997; @cooke2001; @bedford; @aas_2009_paircopula]. 

In the previous example related to height and weight, a vine copula could be used to also model age in relation to height and weight. Like bivariate copulas, vine copulas allow the user to generate random and conditional samples [@cooke_2015_sampling]. However, to draw conditional samples from a vine copula for a specific variable, the vine copula has to be structured in such a way that the order in which the samples are generated draws the variable of interest last, i.e. the sample is conditioned on the preceding samples of other variables. For example, if one wants to generate a conditional sample of height, the samples of age and weight have to be provided first.  Additionally, while it is more common to use copulas for continuous data, such as weight and height, methods have been developed to also allow for discrete data, such as age, to be modelled [@Mitskopoulos2022]. 

`VineCopulas` is a Python package that is able to fit and simulate both bivariate and vine copulas. This package allows for both discrete as well as continuous input data, and can draw conditional samples for any variables of interest with the use of different vine structures (see Figure 1).


![A schematic representation of VineCopulas functionalities, where the lettering refers to the different arrows (Python functions). A) Samples from Table 1 - data, consisting of both continuous and discrete variables (plotted in blue) are transformed into pseudo-observations using their marginal distributions (shown in green). B) A vine copula is fit to the transformed data. Here, the first tree has nodes containing the variables and edges denoting the bivariate dependencies. The edges in the second tree denote the dependency between all variables. C) Using the fitted vine copula, random samples are generated. D) As not every vine copula structure is suitable to generate conditional samples of every variable, due to its inherent sampling order, a vine copula can also be fit conditionally. Here, a vine copula is fit conditionally for variable 1. E) The conditionally fit vine copula is used to draw conditional samples of variable 1 given specific values of variables 2 and 3.\label{fig:schematic}](figure1.png)

# Statement of need

The programming language R is widely known as the most advanced statistical programming language and hence has many well-developed packages for copulas, such as `copula` [@copulapackage],  `VineCopula` [@vinecopulapackage], and `CDVineCopulaConditional` [@Bevacqua]. However, with the open source programming language Python gaining more popularity for statistical programming, there is an increasing interest in Python-based copula packages. Therefore, we have developed the package `VineCopulas`, a pure Python implementation for (vine) copulas. 

`VineCopulas` integrates many of the standard copula package features, including fitting, Probability Density Function (PDFs), and random sample generation for bivariate and vine copulas. This package can also fit the best marginal distributions of the individual variables based on the univariate distributions available in the statistical Python package `SciPy` [@2020SciPy-NMeth]. Furthermore, the `VineCopulas` can compute cumulative distribution functions (CDFs) of bivariate copulas. In addition, the package also enables the user to generate conditional samples, fit vine structures to facilitate specific conditional probabilities and fit as well as simulate discrete data, all of which are unique to have in a single package.

While there are two well-used python copula packages, `copulas` [@inc_copulas], and `pyvinecopulib` [@pyvinecoplib], neither of these packages includes the above-mentioned unique features. Furthermore, `copulas` is mostly suitable for bivariate copulas, and has limited vine copula capabilities, while `pyvinecopulib` is a C++ library with a Python interface, meaning that it is not fully Python-based, and therefore less adaptable for a Python user. Therefore, `VineCopulas` is targeted towards data analysts, researchers and modellers in various fields, who are python users or require functionality specifically for discrete data and conditional sampling.


` VineCopulas` is currently being used in a study on multi-hazards to model the dependencies between different natural hazard intensities. For this study, the ability to generate conditional samples is required to evaluate possible magnitudes of one natural hazard given multiple others e.g., levels of extreme precipitation given specific extreme wind speeds and relative humidity. The capability to also simulate discrete data may be useful for hazards with intensity measures of a discrete nature, such as the Volcanic Explosivity Index (VEI). Applications of this type, are growing in the field of compound and multi-hazard risk research [@Eilander; @Bevacqua_2]. ` VineCopulas` will allow python users to continue this research at a higher dimensionality,  showing the clear need for this package.


# Acknowledgements

This research is carried out in the MYRIAD-EU project. This project has received funding from the European Union’s Horizon 2020 research and innovation programme (Grant Agreement No. 101003276). The work reflects only the author’s view and that the agency is not responsible for any use that may be made of the information it contains. E.E.K. was additionally funded by the European Union’s Horizon 2020 MIRACA project; Grant Agreement No. 101093854.

# References

