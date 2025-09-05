from mmm_src.MMMWorld import MMMWorld
from examples.synths.OscSynth import OscSynth
from mmm_utils.functions import *
from mmm_src.MMMTraits import *

struct ManyOscillators(Representable, Movable, Graphable, Copyable):
    var world_ptr: UnsafePointer[MMMWorld]

    var output: List[Float64]  # Output buffer for audio samples
    # [REVIEW TM] since you're going to make 10, should it just be initialized to that size? or should __init__ have an argument for setting that size? Is it possible to set that size from the .py file?
    var osc_synths: List[OscSynth]  # Instances of the Oscillator


    fn __init__(out self, world_ptr: UnsafePointer[MMMWorld]):
        self.world_ptr = world_ptr
        print("ManyOscillators initialized with num_chans:", self.world_ptr[0].num_chans)

        # [REVIEW TM] Why are they pairs? Because it's likely stereo?
        # initialize the list of oscillator pairs
        self.osc_synths = List[OscSynth]()
        # add 10 pairs to the list
        for _ in range(10):
            # [REVIEW TM] Is OscSynth stereo?
            self.osc_synths.append(OscSynth(self.world_ptr, random_exp_float64(100.0, 1000.0)))  

        # [REVIEW TM] Can't this be initialized to be the right number of channels directly rather than needing a for loop below?
        self.output = List[Float64]()  # Initialize output list

        for _ in range(self.world_ptr[0].num_chans):
            self.output.append(0.0)  # Initialize output list with zeros

    fn __repr__(self) -> String:
        return String("ManyOscillators")

    fn next(mut self: ManyOscillators) -> List[Float64]:

        zero(self.output)  # Clear the output list
        for i in range(len(self.osc_synths)):
            samples = self.osc_synths[i].next()

            # [REVIEW TM] Would it be possible to overload the "+" operator? Does Mojo support overloads like Python's dunders? Or maybe copy SuperCollider with some kind of Out operation?
            mix(self.output, samples)

        '''
        [REVIEW TM] Is there a reason why the above for loop can't be:

        for osc in self.osc_synths:
            mix(self.output, osc.next())
        '''

        return self.output  # Return the combined output sample


