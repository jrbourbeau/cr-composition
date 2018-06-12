
import os
import io
from setuptools import setup, find_packages

VERSION = '0.0.1'

here = os.path.abspath(os.path.dirname(__file__))

def read(path, encoding='utf-8'):
    with io.open(path, encoding=encoding) as f:
        content = f.read()
    return content

def get_install_requirements(path):
    content = read(path)
    requirements = [req for req in content.split("\n")
                    if req != '' and not req.startswith('#')]
    return requirements

INSTALL_REQUIRES = get_install_requirements(os.path.join(here, 'requirements.txt'))

setup(
    name='comptools',
    version=VERSION,
    description='Python tools for cosmic-ray composition analysis',
    author='James Bourbeau',
    packages=find_packages(),
    install_requires=INSTALL_REQUIRES,
    )
