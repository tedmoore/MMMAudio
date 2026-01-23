"""
This example demonstrates how to use MelBands
"""

from mmm_python import *
ma = MMMAudio(128, graph_name="MelBandsExample", package_name="examples")
ma.start_audio()

ma.send_float("multiplier",600.0)

ma.stop_audio()