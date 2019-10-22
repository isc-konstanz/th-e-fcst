#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    theforecast
    ~~~~~
    
    
"""
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

VERSION = '0.1.0'

DESCRIPTION = 'TH-E Forecast provides a set of functions to predict timeseries.'

NAME = 'theforecast'
AUTHOR = 'ISC Konstanz'
MAINTAINER_EMAIL = 'steffen.friedriszik@isc-konstanz.de'

INSTALL_REQUIRES = ['numpy >= 1.10.1',
                    'pandas >= 0.14.1',
                    'mysql-connector',
                    'ortools']

PACKAGES = ['theforecast']

SCRIPTS = ['bin/th-e-forecast']

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    author=AUTHOR,
    author_email=MAINTAINER_EMAIL,
    packages=PACKAGES,
    install_requires=INSTALL_REQUIRES,
    scripts=SCRIPTS,
)
