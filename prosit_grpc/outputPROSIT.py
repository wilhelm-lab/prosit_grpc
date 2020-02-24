from . import __constants__ as C  # For constants
from . import __utils__ as U
import numpy as np


# Output
class PROSIToutput:
    def __init__(self):
        self.spectrum = PROSITspectrum()
        self.irt = PROSITirt()
        self.proteotypicity = PROSITproteotypicity()

    def prepare_output(self, charges_array, sequences_lengths):
        # prepare spectrum
        self.spectrum.prepare_spectrum(charges_array=charges_array,
                                       sequences_lengths=sequences_lengths)
        # prepare irt
        self.irt.normalize()

        # prepare proteotypicity


class PROSITproteotypicity:
    def __init__(self):
        self.raw = None


class PROSITirt:
    def __init__(self):
        self.raw = None
        self.normalized = None

    def normalize(self):
        self.normalized = [i * 43.39373 + 56.35363441 for i in self.raw]


class PROSITspectrum:
    def __init__(self):
        self.intensity = PROSITintensity()
        self.mz = PROSITfragmentmz()
        self.annotation = PROSITfragmentannotation()

        self.mask = None
        self.filter = None

    @staticmethod
    def create_masking(charges_array, sequences_lengths):
        """
        assume reshaped output of prosit, shape sould be (num_seq, 174)
        set filtered output where not allowed positions are set to -1
        prosit output has the form:
        y1+1     y1+2 y1+3     b1+1     b1+2 b1+3     y2+1     y2+2 y2+3     b2+1     b2+2 b2+3
        if charge >= 3: all allowed
        if charge == 2: all +3 invalid
        if charge == 1: all +2 & +3 invalid
        """

        assert len(charges_array) == len(sequences_lengths)

        mask = []

        for i in range(len(charges_array)):
            charge_one_hot = charges_array[i]
            len_seq = sequences_lengths[i]
            m = []

            # filter according to peptide charge
            if np.array_equal(charge_one_hot, [1, 0, 0, 0, 0, 0]):
                invalid_indexes = [(x * 3 + 1) for x in range((C.SEQ_LEN - 1) * 2)] + [(x * 3 + 2) for x in
                                                                                       range((C.SEQ_LEN - 1) * 2)]
                m.extend([invalid_indexes])

            elif np.array_equal(charge_one_hot, [0, 1, 0, 0, 0, 0]):
                invalid_indexes = [x * 3 + 2 for x in range((C.SEQ_LEN - 1) * 2)]
                m.extend([invalid_indexes])

            if len_seq < C.SEQ_LEN:
                m.extend(range(start=(len_seq - 1) * 6,stop=C.SEQ_LEN))



        return mask

    def apply_masking(self):
        self.intensity.masked = np.multiply(self.intensity.raw, self.mask)
        self.mz.masked = np.multiply(self.mz.raw, self.mask)

        self.annotation.masked_type = np.multiply(self.annotation.raw_type, self.mask)
        self.annotation.masked_charge = np.multiply(self.annotation.raw_charge, self.mask)
        self.annotation.masked_number = np.multiply(self.annotation.raw_number, self.mask)

    def create_filter(self):
        self.filter = self.intensity.normalized != -1

    def apply_filter(self):
        self.intensity.filtered = self.intensity.normalized[self.filter]
        self.mz.filtered = self.mz.masked[self.filter]
        self.annotation.filtered = self.annotation.masked[self.filter]

    def prepare_spectrum(self, charges_array, sequences_lengths):

        self.mask = PROSITspectrum.create_masking(charges_array=charges_array,
                                                  sequences_lengths=sequences_lengths)
        self.apply_masking()
        self.intensity.normalize()

        self.create_filter()
        self.apply_filter()


class PROSITintensity:
    def __init__(self):
        self.raw = None
        self.masked = None
        self.normalized = None
        self.filtered = None

    def normalize(self):
        self.normalized = U.normalize_intensities(self.masked)
        self.normalized[self.normalized < 0] = -1


class PROSITfragmentmz:
    def __init__(self):
        self.raw = None
        self.masked = None
        self.filtered = None


class PROSITfragmentannotation:
    def __init__(self):
        self.raw_charge = None
        self.raw_number = None
        self.raw_type = None

        self.masked_charge = None
        self.masked_number = None
        self.masked_type = None

        self.filtered_charge = None
        self.filtered_number = None
        self.filtered_type = None
