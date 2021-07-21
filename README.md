# Quick-Image-Separations

A module for Python 3.6 designed to quickly estimate image separations of lensed sources by extracting a 5-sigma mask of an input image and determining a best-fit circle using [Orthogonal Distance Regression](https://docs.scipy.org/doc/scipy/reference/odr.html).

This has been developed for application to continuum images of lensed sources pulled from the [ALMA Science Archive](https://almascience.nrao.edu/aq/)

This requires the latest versions of numpy, scipy, astropy, matplotlib, and mpl_toolkits.

Currently, to install simply place 'quims.py' into the same directory as your current script and run 'import quims'.
