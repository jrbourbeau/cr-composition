
from setuptools import setup

VERSION = '0.0.1'

with open('requirements.txt') as fid:
    INSTALL_REQUIRES = [l.strip() for l in fid.readlines() if l]

setup(
    name='comptools',
    version=VERSION,
    description='Python tools for cosmic-ray composition analysis',
    author='James Bourbeau',
    packages=['comptools'],
    install_requires=INSTALL_REQUIRES,
)
