import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from mmm_python import *

if __name__ == "__main__":
    # instantiate and load the graph
    m = MMMAudio(128, graph_name="TestToPython", package_name="examples.tests")

    # m.register_callback("pitch", lambda v: print(f"pitch: {v}"))

    # for i in range(200):
    #     m.register_callback("val_" + str(i), lambda v, i=i: print(f"val_{i}: {v}"))

    m.start_audio()

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        m.stop_audio()
        m.stop_process()
