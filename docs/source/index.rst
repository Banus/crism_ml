CRISM Machine Learning toolkit
==============================

In this package we demonstrate the utility of machine learning in two essential
CRISM analysis tasks: nonlinear noise removal and mineral classification.
We propose a simple yet effective hierarchical Bayesian model (HBM) for the
estimation of the distributions of spectral patterns and we extensively
validate this model for mineral classification on several test images.

The package implemented scripts, documentation illustrating use cases, and
pixel-scale training data collected from dozens of well-characterized CRISM
images. The goal of this new toolkit is to provide advanced and effective
processing tools and improve the ability of the planetary community to map
compositional units in remote sensing data quickly, accurately, and at scale.

Project page: `CRISM ML toolkit`_.

.. _CRISM ML toolkit: http://cs.iupui.edu/~mdundar/CRISM.htm

.. autosummary::
   :toctree: generate
   :template: custom-module-template.rst

   crism_ml
   crism_ml.io
   crism_ml.lab
   crism_ml.models
   crism_ml.plot
   crism_ml.preprocessing
   crism_ml.train

Installation
============
The code requires Python 3.7 or newer and it runs on Windows and Linux.
You can install an environment with all the dependencies with:

.. code-block:: bash

   conda env create -f environment.yml

A ``requirements.txt`` file is also available if you do not have Anaconda; run
this in your virtual environment:

.. code-block:: bash

   pip install -r requirements.txt

To use the package from any path, install it in your environment with the
following command from the project root:

.. code-block:: bash

   pip install -e .

This command will also take care of the project dependencies if you didn't
install them and add the project to the python packages in editable mode.

Training script
===============

.. argparse::
   :module: crism_ml.train
   :func: get_parser
   :prog: train

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
