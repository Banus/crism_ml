"""Package setup."""
from setuptools import setup, find_packages

setup(
    name='crism_ml',
    version='0.0.1',
    # url='https://github.com/mypackage.git',
    author='Emanuele Plebani',
    author_email='eplebani@iu.edu',
    description='Hyperspectral mineral classification',
    packages=find_packages(),
    install_requires=[
        'joblib >=0.14',
        'matplotlib >= 3.1',
        'numpy >= 1.18',
        'scipy >= 1.4',
        'spectral >= 0.19',
        'Bottleneck>=1.3',
        'mat73 >= 0.46'],
)
