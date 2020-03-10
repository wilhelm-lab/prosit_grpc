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
        self.irt.prepare_irt()

        # prepare proteotypicity
        # nothing to prepare proteotypicity is only reported raw

class PROSITproteotypicity:
    def __init__(self):
        self.raw = None


class PROSITirt:
    def __init__(self):
        self.raw = None
        self.normalized = None

    def normalize(self):
        self.normalized = [i * 43.39373 + 56.35363441 for i in self.raw]
        self.normalized = np.array(self.normalized)
        self.normalized.shape = self.raw.shape

    def prepare_irt(self):
        if type(self.raw) == np.ndarray:
            self.normalize()

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

        mask = np.ones(shape=(len(charges_array), C.VEC_LENGTH), dtype=np.int32)

        for i in range(len(charges_array)):
            charge_one_hot = charges_array[i]
            len_seq = sequences_lengths[i]
            m = mask[i]

            # filter according to peptide charge
            if np.array_equal(charge_one_hot, [1, 0, 0, 0, 0, 0]):
                invalid_indexes = [(x * 3 + 1) for x in range((C.SEQ_LEN - 1) * 2)] + [(x * 3 + 2) for x in
                                                                                       range((C.SEQ_LEN - 1) * 2)]
                m[invalid_indexes] = -1

            elif np.array_equal(charge_one_hot, [0, 1, 0, 0, 0, 0]):
                invalid_indexes = [x * 3 + 2 for x in range((C.SEQ_LEN - 1) * 2)]
                m[invalid_indexes] = -1

            if len_seq < C.SEQ_LEN:
                invalid_indexes = range((len_seq - 1) * 6, C.VEC_LENGTH)
                m[invalid_indexes] = -1

        return mask

    def apply_masking(self):

        invalid_indices = self.mask == -1

        self.intensity.masked = np.copy(self.intensity.raw)
        self.intensity.masked[invalid_indices] = -1

        self.mz.masked = np.copy(self.mz.raw)
        self.mz.masked[invalid_indices] = -1

        self.annotation.masked_charge = np.copy(self.annotation.raw_charge)
        self.annotation.masked_charge[invalid_indices] = -1
        self.annotation.masked_number = np.copy(self.annotation.raw_number)
        self.annotation.masked_number[invalid_indices] = -1

        self.annotation.masked_type = np.copy(self.annotation.raw_type)
        self.annotation.masked_type[invalid_indices] = None

    def create_filter(self):
        self.filter = self.intensity.normalized != -1

    def apply_filter(self):

        self.intensity.filtered = []
        self.mz.filtered = []

        self.annotation.filtered_charge = []
        self.annotation.filtered_number = []
        self.annotation.filtered_type = []


        for i in range(len(self.filter)):
            self.intensity.filtered.append(self.intensity.normalized[i][self.filter[i]])
            self.mz.filtered.append(self.mz.masked[i][self.filter[i]])

            self.annotation.filtered_number.append(self.annotation.masked_number[i][self.filter[i]])
            self.annotation.filtered_charge.append(self.annotation.masked_charge[i][self.filter[i]])
            self.annotation.filtered_type.append(self.annotation.masked_type[i][self.filter[i]])

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
        self.normalized[self.normalized < 0] = 0
        self.normalized[self.masked == -1] = -1


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
