#!/usr/bin/env python

# Copyright (c) 2017 by the parties listed in the AUTHORS
# file.  All rights reserved.  Use of this source code is governed
# by a BSD-style license that can be found in the LICENSE file.

"""map_tools setup.py
"""

import glob
import os
import re

from setuptools import setup


def get_version():
    """Get map_tools version from _version.py
    """
    ver = 'unknown'
    if os.path.isfile('map_tools/_version.py'):
        with open('map_tools/_version.py', 'r') as verfile:
            for line in verfile.readlines():
                matched = re.match("__version__ = '(.*)'", line)
                if matched:
                    ver = matched.group(1)
    return ver


CURRENT_VERSION = get_version()
SCRIPTS = glob.glob('scripts/*.py')


setup(
    name='map_tools',
    provides='map_tools',
    version=CURRENT_VERSION,
    description='Simple map tools',
    author='Reijo Keskitalo',
    url='https://github.com/hpc4cmb/map_tools',
    packages=['map_tools'],
    scripts=SCRIPTS,
    license='BSD',
    requires=['Python (>3.4.0)'],
)
