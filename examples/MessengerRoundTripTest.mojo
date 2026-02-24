from mmm_audio import *

struct MessengerRoundTripTest(Movable, Copyable):
    var world: World
    var m: Messenger
    var float_val: Float64
    var floats_val: List[Float64]
    var int_val: Int
    var ints_val: List[Int]
    var bool_val: Bool
    var str: String
    var strs: List[String]

    fn __init__(out self, world: World):
        self.world = world
        self.m = Messenger(self.world)
        self.float_val = 0.0
        self.floats_val = List[Float64]()
        self.int_val = 0
        self.ints_val = List[Int]()
        self.bool_val = False
        self.str = ""
        self.strs = List[String]()

    fn next(mut self) -> SIMD[DType.float64, 2]:
        
        if self.m.notify_update(self.float_val,"float"):
            print("mojo received float: ",self.float_val)
            self.m.to_python("float_return", self.float_val + 1.0)
        
        if self.m.notify_update(self.floats_val,"floats"):
            print("mojo received floats: ",self.floats_val)
            self.m.to_python("floats_return", [x + 1.0 for x in self.floats_val])

        if self.m.notify_update(self.int_val,"int"):
            print("mojo received int: ",self.int_val)
            self.m.to_python("int_return", self.int_val + 1)
        
        if self.m.notify_update(self.ints_val,"ints"):
            print("mojo received ints: ",self.ints_val)
            self.m.to_python("ints_return", [x + 1 for x in self.ints_val])
        
        if self.m.notify_update(self.bool_val,"bool"):
            print("mojo received bool: ",self.bool_val)
            self.m.to_python("bool_return", not self.bool_val)
        
        if self.m.notify_trig("trig"):
            print("mojo received trig: 'trig'")
            self.m.to_python("trig_return")
        
        if self.m.notify_update(self.str,"str"):
            print("mojo received str: ",self.str)
            self.m.to_python("str_return", self.str + "_return")
        
        if self.m.notify_update(self.strs,"strs"):
            print("mojo received strs: ",self.strs)
            self.m.to_python("strs_return", [s + "_return" for s in self.strs])
        
        return SIMD[DType.float64, 2](0.0, 0.0)