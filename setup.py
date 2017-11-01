'''
comptools
Author: James Bourbeau
'''

from setuptools import setup

VERSION = '0.0.1'

install_requires = ['numpy', 'scipy', 'pandas', 'matplotlib', 'seaborn',
                    'scikit-learn', 'mlxtend', 'xgboost', 'pycondor',
                    'tables', 'PyMySQL', 'healpy', 'jupyter', 'ipython',
                    'pyprind', 'pytest', 'dask[complete]']
setup(
    name='comptools',
    version=VERSION,
    description='Python tools for cosmic-ray composition analysis',
    author='James Bourbeau',
    packages=['comptools'],
    install_requires=install_requires
)
