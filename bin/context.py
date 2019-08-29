"""
File: context.py
Author: Bastien
Email: bastien.vieuble@gmail.com
Github: https://github.com/bvieuble
Description: This file contains the context where should be executed your
scripts. Import it at the beginning of your bin/file to execute it as if you
were at the root of the repo.
"""


import os
import sys


thisdir = os.path.dirname(__file__)
rootdir = os.path.join(thisdir, '..')

if rootdir not in sys.path:
    sys.path.insert(0, rootdir)
    os.chdir(rootdir)
