
from setuptools import setup

VERSION = '0.0.1'

with open('requirements.txt', 'r') as f:
    INSTALL_REQUIRES = [l.strip() for l in f.readlines() if l]
    print('INSTALL_REQUIRES = {}'.format(INSTALL_REQUIRES))

setup(
    name='comptools',
    version=VERSION,
    description='Python tools for cosmic-ray composition analysis',
    author='James Bourbeau',
    packages=['comptools'],
    install_requires=INSTALL_REQUIRES
)
