from mmm_python.MMMAudio import *
from mmm_python.python_utils import *
from mmm_python.Patterns import *
import asyncio
from mmm_python.OSCServer import *
from mmm_python.Scheduler import *

import os

SKIP_GUI = os.getenv("SKIP_GUI", "0") == "1"

if not SKIP_GUI:
    from mmm_python.GUI import *

from mmm_python.hid_devices import *