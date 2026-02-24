# import sys
# import time
# from pathlib import Path
# sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from mmm_python import *
m = MMMAudio(128, graph_name="TestToPython", package_name="examples.tests")
m.register_callback("pitch", lambda args: print(f"pitch: {args}"))
# for i in range(200):
#     m.register_callback(f"val_{i}", lambda args, i=i: print(f"val_{i}: {args}"))
m.start_audio()

m.stop_audio()