
from mmm_audio import *

struct TestToPython(Movable, Copyable):
    var world: World
    var m: Messenger
    var yin: BufferedInput[YIN[1024],1024,512]
    var buf: Buffer
    var play: Play

    fn __init__(out self, world: World):
        self.world = world
        self.m = Messenger(self.world)
        self.buf = Buffer.load("resources/Shiverer.wav")
        self.play = Play(self.world)
        yin = YIN[1024](self.world)
        self.yin = BufferedInput[YIN[1024],1024,512](self.world,yin^)

    fn next(mut self) -> SIMD[DType.float64, 2]:

        sig = self.play.next(self.buf)
        self.yin.next(sig)
        
        self.m.to_python("pitch", self.yin.process.pitch)

        return SIMD[DType.float64, 2](sig, sig)