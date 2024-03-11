
---
title: 'VineCopulas: .....'
tags:
  - Python
  - copula
  - statistics
authors:
  - name: Judith N. Claassen
    orcid: 0009-0000-2979-1297
    affiliation: 1
  - name: Wiebke S. Jäger
    orcid: 0009-0007-8628-6060
    affiliation: 1
  - name: Elco E. Koks
    orcid: 0000-0002-4953-4527
    affiliation: 1
affiliations:
 - name: Institute for Environmental Studies, Vrije Universiteit Amsterdam, Amsterdam, The Netherlands
   index: 1
date: 7 March 2024
bibliography: paper.bib
---

# Summary
A copula method can be used used to describe the dependency structure between several random variables. [*add brief example here*] Copula-methods are used widely in various research fields across different disciplines, such as finance and hydrology. While some other multivariate distributions, for instance a multivariate normal distribution, allow for a highly symmetric dependency structure with the same univariate and multivariate marginal distributions, copulas can model the joint distribution of multiple random variables separately from their marginal distribution [@Czado2022].

Copulas allow for random samples of the data to be generated, as well as conditional samples. For example, if a copula has been fit between people's height and weight, this copula can create random correlated samples of both variables as well as conditional samples, e.g., samples of weight given a specific height.

Although bivariate copulas are an excellent tool to model dependencies between two variables [*I think a small description of bivariate is required*], there are only a limited number of copulas capable of modeling larger multivariate datasets [*you mean limited number of bivariate copulas, or do you refer to other type of copulas again?*], such as the Gaussian and Student-t copula. However, when modeling the dependencies between a large amount of different variables, a more flexible multivariate modeling tool may be required that does not assume a single copula to capture all the individual dependencies. To this end, vine copulas have been proposed as a method to construct a multivariate model with the use of bivariate copulas as building blocks [@Joe2014]. 

For example, in addition to weight and height, a vine copula can also model age. Like bivariate copulas, vine copulas allow to generate random and conditional samples. However, to draw conditional samples from a vine copula for a specific variable, the vine copula has to be structured in such a way that the order in which the samples are generated draws the variable of interest last, i.e. the sample is conditioned on the preceding samples of other variables. For example, if one wants to generate a conditional sample of height, the samples of age and weight have to be provided first.  Additionally, while it is more common to use copulas for continuous data, such as weight and height, methods have been developed to also allow for discrete data, such as age, to be modeled [@Mitskopoulos2022]. 

`Vinecopulas` is a Python package that is able to fit and simulate both bivariate and vine copulas. This package allows for both discrete as well as continuous input data, and can draw conditional samples for any variable of interest with the use of different vine structures.

# Statement of need

The programming language R is widely known as the most advanced statistical programming language and hence has many excellent packages for copulas, such as `copula` [@copulapackage],  `VineCopula` [@vinecopulapackage], and `CDVineCopulaConditional` [@Bevacqua]. However, with open source programming language Python gaining more popularity [*I think python is already very popular, perhaps you mean for statistical programming*], there is an increasing interest in Python-based copula packages. Therefore, we have developed the package `Vinecopulas`, a pure Python implementation for (vine) copulas. 

`Vinecopulas` integrates many of the standard copula package features, including fitting, conditional distribution functions (CDFs), Probability Density Function (PDFs), and random sampling generation, for both vine and bivariate copulas, into a single package. In addition, the package also enables to generate conditional samples, fit vine structures to facilitate specific conditional probabilites and fit as well as simulate discrete data, all of which are unique to have in a single package.
While there are two excellent python packages, `copulas` [@inc_copulas], and `pyvinecopulib` [@pyvinecoplib], neither of these packages includes the above-mentioned unique features. furthermore, `copulas` is mostly suitable for bivariate copulas, and has limited vine copula capabilities, while `pyvinecopulib` is C++ library with a Python interface, hence is not fully Python-based. 

_Mentions (if applicable) of any ongoing research projects using the software or recent scholarly publications enabled by it._

` Vinecopulas` is currently being used in a study on multi-hazards to model the dependencies between different natural hazard intensities. For this study, the ability to generate conditional samples is required to evaluate possible magnitudes of one natural hazard given another e.g., levels of extreme precipitation given specific extreme wind speeds. The capability to also simulate discrete data may be usual for hazards with intensity measures of a discrete nature, such as the volcanic explosivity index (VEI).

# Acknowledgements

This research is carried out in the MYRIAD-EU project. This project has received funding from the European Union’s Horizon 2020 research and innovation programme (Grant Agreement No. 101003276). The work reflects only the author’s view and that the agency is not responsible for any use that may be made of the information it contains. E.E.K.was additionally funded by the e European Union’s Horizon 2020 MIRACA project; Grant Agreement No. 101093854.

# References

