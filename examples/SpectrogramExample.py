import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from mmm_python import *

if __name__ == "__main__":
    m = MMMAudio(128, graph_name="SpectrogramExample", package_name="examples")
    m.register_callback("mags", lambda args: print(f"mags: {args}"))
    m.start_audio()