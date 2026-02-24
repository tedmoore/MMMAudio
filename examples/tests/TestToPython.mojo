
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
        self.m.to_python("val_0", sig[0])
        self.m.to_python("val_1", sig[0] + 1)
        self.m.to_python("val_2", sig[0] + 2)
        self.m.to_python("val_3", sig[0] + 3)
        self.m.to_python("val_4", sig[0] + 4)
        self.m.to_python("val_5", sig[0] + 5)
        self.m.to_python("val_6", sig[0] + 6)
        self.m.to_python("val_7", sig[0] + 7)
        self.m.to_python("val_8", sig[0] + 8)
        self.m.to_python("val_9", sig[0] + 9)
        self.m.to_python("val_10", sig[0] + 10)
        self.m.to_python("val_11", sig[0] + 11)
        self.m.to_python("val_12", sig[0] + 12)
        self.m.to_python("val_13", sig[0] + 13)
        self.m.to_python("val_14", sig[0] + 14)
        self.m.to_python("val_15", sig[0] + 15)
        self.m.to_python("val_16", sig[0] + 16)
        self.m.to_python("val_17", sig[0] + 17)
        self.m.to_python("val_18", sig[0] + 18)
        self.m.to_python("val_19", sig[0] + 19)
        self.m.to_python("val_20", sig[0] + 20)
        self.m.to_python("val_21", sig[0] + 21)
        self.m.to_python("val_22", sig[0] + 22)
        self.m.to_python("val_23", sig[0] + 23)
        self.m.to_python("val_24", sig[0] + 24)
        self.m.to_python("val_25", sig[0] + 25)
        self.m.to_python("val_26", sig[0] + 26)
        self.m.to_python("val_27", sig[0] + 27)
        self.m.to_python("val_28", sig[0] + 28)
        self.m.to_python("val_29", sig[0] + 29)
        self.m.to_python("val_30", sig[0] + 30)
        self.m.to_python("val_31", sig[0] + 31)
        self.m.to_python("val_32", sig[0] + 32)
        self.m.to_python("val_33", sig[0] + 33)
        self.m.to_python("val_34", sig[0] + 34)
        self.m.to_python("val_35", sig[0] + 35)
        self.m.to_python("val_36", sig[0] + 36)
        self.m.to_python("val_37", sig[0] + 37)
        self.m.to_python("val_38", sig[0] + 38)
        self.m.to_python("val_39", sig[0] + 39)
        self.m.to_python("val_40", sig[0] + 40)
        self.m.to_python("val_41", sig[0] + 41)
        self.m.to_python("val_42", sig[0] + 42)
        self.m.to_python("val_43", sig[0] + 43)
        self.m.to_python("val_44", sig[0] + 44)
        self.m.to_python("val_45", sig[0] + 45)
        self.m.to_python("val_46", sig[0] + 46)
        self.m.to_python("val_47", sig[0] + 47)
        self.m.to_python("val_48", sig[0] + 48)
        self.m.to_python("val_49", sig[0] + 49)

        return SIMD[DType.float64, 2](sig, sig)