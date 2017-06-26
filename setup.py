'''
comptools
Author: James Bourbeau
'''

from setuptools import setup

VERSION = '0.0.1'

setup(
    name='comptools',
    version=VERSION,
    description='Python tools for cosmic-ray composition analysis',
    author='James Bourbeau',
    author_email='jbourbeau@wisc.edu',
    packages=['comptools'],
    install_requires=['numpy', 'scipy', 'pandas', 'matplotlib', 'seaborn',
        'scikit-learn', 'mlxtend', 'xgboost', 'pycondor', 'tables', 'PyMySQL',
        'healpy', 'watermark', 'pyprind', 'pytest']
)
