import time

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from mmm_python import *

def float_return(fl,d):
    print(f"python received float: {fl}")

def main():
    m = MMMAudio(128, graph_name="MessengerRoundTripTest", package_name="examples")
    m.start_audio()

    received_values = {}

    m.register_callback("float_return",float_return)
    m.register_callback("floats_return", lambda args: received_values.update({"floats": args}))
    m.register_callback("int_return", lambda args: received_values.update({"int": args}))
    m.register_callback("ints_return", lambda args: received_values.update({"ints": args}))
    m.register_callback("bool_return", lambda args: received_values.update({"bool": args}))
    m.register_callback("str_return", lambda args: received_values.update({"str": args}))
    m.register_callback("strs_return", lambda args: received_values.update({"strs": args}))
    m.register_callback("trig_return", lambda args: received_values.update({"trig": args}))

    m.send_float("float",2.1415)
    m.send_floats("floats",[1.0,2.0,3.0])
    m.send_int("int",42)
    m.send_ints("ints",[100, 200, 300])
    m.send_bool("bool",True)
    m.send_string("str","hello")
    m.send_strings("strs",["foo", "bar", "baz"])
    m.send_trig("trig")

    time.sleep(10)  # Give some time for the audio thread to start

    expected_values = {
        "float": 3.1415,
        "floats": np.array([2.0, 3.0, 4.0]),
        "int": 43,
        "ints": np.array([101, 201, 301]),
        "bool": False,
        "str": "hello_return",
        "strs": ["foo_return", "bar_return", "baz_return"],
        "trig": None
    }

    print("received values: ",received_values)

    for k,v in received_values.items():
        print(f"key: {k}\t\texpected: {expected_values[k]}\t\treceived: {v}")
    
    m.stop_audio()
    m.stop_process()

if __name__ == "__main__":
    main()