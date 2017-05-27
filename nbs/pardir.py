"""Allow imports from parent directory."""

import inspect
import os
import sys

def pardir():
    sourcefile = os.path.abspath(inspect.getsourcefile(lambda: 0))
    current_dir = os.path.dirname(sourcefile)
    parent_dir = os.path.join(current_dir, os.path.pardir)
    sys.path.insert(0, parent_dir)
