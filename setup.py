#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    th-e-fcst
    ~~~~~~~~~
    
    
"""
from os import path
from setuptools import setup, find_namespace_packages

here = path.abspath(path.dirname(__file__))
info = {}
with open(path.join("th_e_fcst", "_version.py")) as f:
    exec(f.read(), info)

VERSION = info['__version__']

DESCRIPTION = 'TH-E Forecast provides a set of functions to predict timeseries.'

# Get the long description from the README file
# with open(path.join(here, 'README.md')) as f:
#     README = f.read()

NAME = 'th-e-fcst'
LICENSE = 'LGPLv3'
AUTHOR = 'ISC Konstanz'
MAINTAINER_EMAIL = 'adrian.minde@isc-konstanz.de'
URL = 'http://gitlab.isc-konstanz.de/systems/systems/th-e-fcst'

INSTALL_REQUIRES = [
    'keras >= 2.6.0',
    'tensorflow >= 2.6.0',
    'pvsys @ git+https://github.com/isc-konstanz/pvsys.git@master'
]

EXTRAS_REQUIRE = {
    'eval': ['scisys[excel,plot] @ git+https://github.com/isc-konstanz/scisys.git@master']
}

SCRIPTS = ['bin/th-e-fcst']

PACKAGES = find_namespace_packages(include=['th_e_fcst*'])

SETUPTOOLS_KWARGS = {
    'zip_safe': False,
    'include_package_data': True
}

setup(
    name=NAME,
    version=VERSION,
    license=LICENSE,
    description=DESCRIPTION,
    # long_description=README,
    author=AUTHOR,
    author_email=MAINTAINER_EMAIL,
    packages=PACKAGES,
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    scripts=SCRIPTS,
    **SETUPTOOLS_KWARGS
)
