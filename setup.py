'''
comptools
Author: James Bourbeau
'''

from setuptools import setup
import comptools

VERSION = comptools.__version__

setup(
    name='comptools',
    version=VERSION,
    description='Python tools for cosmic-ray composition analysis',
    author='James Bourbeau',
    author_email='jbourbeau@wisc.edu',
    packages=['comptools'],
    install_requires=['numpy', 'scipy', 'pandas', 'matplotlib', 'seaborn', 'scikit-learn', 'mlxtend', 'xgboost', 'pycondor', 'tables']
)
