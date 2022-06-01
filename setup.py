#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    th-e-fcst
    ~~~~~~~~~
    
    
"""
from os import path

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

here = path.abspath(path.dirname(__file__))
info = {}
with open(path.join("th_e_fcst", "_version.py")) as f: exec(f.read(), info)

VERSION = info['__version__']

DESCRIPTION = 'TH-E Forecast provides a set of functions to predict timeseries.'

# Get the long description from the README file
# with open(path.join(here, 'README.md')) as f:
#     README = f.read()

NAME = 'th-e-fcst'
# LICENSE = 'LGPLv3'
AUTHOR = 'ISC Konstanz'
MAINTAINER_EMAIL = 'steffen.friedriszik@isc-konstanz.de'
URL = 'http://gitlab.isc-konstanz.de/systems/systems/th-e-fcst'

INSTALL_REQUIRES = ['numpy == 1.22.*',
                    'pandas == 1.4.*',
                    'keras == 2.6.*',
                    'tensorflow == 2.6.*',
                    'th_e_core >= 0.4.2',
                    'th_e_yield >= 0.1.2']

SCRIPTS = ['bin/th-e-fcst']

PACKAGES = ['th_e_fcst']

SETUPTOOLS_KWARGS = {
    'zip_safe': False,
    'include_package_data': True
}

setup(
    name=NAME,
    version=VERSION,
    # license=LICENSE,
    description=DESCRIPTION,
    # long_description=README,
    author=AUTHOR,
    author_email=MAINTAINER_EMAIL,
    packages=PACKAGES,
    install_requires=INSTALL_REQUIRES,
    scripts=SCRIPTS,
)
