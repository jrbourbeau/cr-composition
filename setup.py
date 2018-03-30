
from setuptools import setup

VERSION = '0.0.1'

with open('requirements.txt', 'r') as f:
    INSTALL_REQUIRES = [l.strip() for l in f.readlines() if l]

for idx, requirement in enumerate(INSTALL_REQUIRES):
    if 'pyunfold' in requirement:
        INSTALL_REQUIRES[idx] = 'pyunfold'

setup(
    name='comptools',
    version=VERSION,
    description='Python tools for cosmic-ray composition analysis',
    author='James Bourbeau',
    packages=['comptools'],
    install_requires=INSTALL_REQUIRES,
    dependency_links=[
        'git+https://github.com/jrbourbeau/pyunfold.git@master#egg=pyunfold-0.0.1.dev0',
        ],
    )
