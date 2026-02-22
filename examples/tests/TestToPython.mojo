
from mmm_audio import *

struct TestToPython(Movable, Copyable):
    var world: World
    var m: Messenger
    var yin: BufferedInput[YIN[1024],1024,512]
    var buf: Buffer
    var play: Play
    var vals: List[Float64]
    var impulse: Impulse[]

    fn __init__(out self, world: World):
        self.world = world
        self.m = Messenger(self.world)
        self.buf = Buffer.load("resources/Shiverer.wav")
        self.play = Play(self.world)
        yin = YIN[1024](self.world)
        self.yin = BufferedInput[YIN[1024],1024,512](self.world,yin^)
        self.vals = List[Float64](length=200,fill=0.0)
        self.impulse = Impulse(self.world)

    fn next(mut self) -> SIMD[DType.float64, 2]:

        sig = self.play.next(self.buf)
        self.yin.next(sig)

        for i in range(200):
            self.vals[i] = random_float64()
        
        if self.impulse.next_bool(10):
            self.m.to_python("pitch", self.yin.process.pitch)    
            # for i in range(200):
            #     self.m.to_python("val_" + String(i), self.vals[i])

        return SIMD[DType.float64, 2](sig, sig)