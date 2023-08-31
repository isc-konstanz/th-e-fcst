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
URL = 'http://gitlab.isc-konstanz.de/systems/th-e-fcst'

INSTALL_REQUIRES = [
    'holidays',
    'keras < 2.11',
    'tensorflow < 2.11',
    'pvsys @ git+https://github.com/isc-konstanz/pvsys.git@v0.2.8'
]

EXTRAS_REQUIRE = {
    'eval': [
        'tqdm',
        'scisys[excel,plot] @ git+https://github.com/isc-konstanz/scisys.git@v0.2.8'
    ]
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
