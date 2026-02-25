from mmm_audio import *

struct Spectrogram(FFTProcessable):
    var world: World
    var m: Messenger

    fn __init__(out self, world: World):
        self.world = world
        self.m = Messenger(world)

    fn next_frame(mut self, mut mags: List[Float64], mut freqs: List[Float64]):
        self.m.reply_once("mags", mags)

struct SpectrogramExample(Movable, Copyable):
    var world: World
    var buf: Buffer
    var play: Play
    var fftproces: FFTProcess[Spectrogram]

    fn __init__(out self, world: World):
        self.world = world
        self.buf = Buffer.load("resources/Shiverer.wav")
        self.play = Play(world)
        self.fftproces = FFTProcess(world, Spectrogram(world))

    fn next(mut self) -> MFloat[2]:
        sig = self.play.next(self.buf)
        _ = self.fftproces.next(sig)
        return sig