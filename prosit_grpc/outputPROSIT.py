from . import __constants__ as C  # For constants
import numpy as np

# Output
class PROSIToutput:
    def __init__(self, intensity_raw, irt_raw, proteotypicity_raw):
        self.spectrum = PROSITspectrum(intensity_raw)
        self.irt = PROSITirt(irt_raw)
        self.proteotypicity = PROSITproteotypicity(proteotypicity_raw)

class PROSITproteotypicity:
    pass

class PROSITirt:
    pass

class PROSITspectrum:
    def __init__(self):
        self.intensity = PROSITintensity()
        self.mz = PROSITfragmentmz()
        self.annotation = PROSITfragmentannotation()

class PROSITintensity:
    def __init__(self):
        self.raw = np.ndarray(shape=(0,C.VEC_LENGTH))

class PROSITfragmentmz:
    def __init__(self):
        self.raw = np.ndarray(shape=(0,C.VEC_LENGTH))

class PROSITfragmentannotation:
    def __init__(self, ref_parent):
        self.raw = np.ndarray(shape=(0,C.VEC_LENGTH))
