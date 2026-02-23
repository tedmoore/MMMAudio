# import sys
# import time
# from pathlib import Path
# sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from mmm_python import *
m = MMMAudio(128, graph_name="TestToPython", package_name="examples.tests")
m.start_audio()
