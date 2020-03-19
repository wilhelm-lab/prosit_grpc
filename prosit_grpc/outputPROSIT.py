from . import __constants__ as C  # For constants
from . import __utils__ as U
import numpy as np


# Output
class PROSIToutput:
    def __init__(self, pred_intensity, pred_irt, pred_proteotyp,sequences_array, charges_array):
        
        if pred_intensity is not None:
            self.spectrum = PROSITspectrum(
                pred_intensity=pred_intensity,
                sequences_array=sequences_array,
                charges_array=charges_array)
        else:
            self.spectrum = None

        if pred_irt is not None:
            self.irt = PROSITirt(pred_irt=pred_irt)
        else:
            self.irt = None

        if pred_proteotyp is not None:
            self.proteotypicity = PROSITproteotypicity(pred_proteotyp=pred_proteotyp)
        else:
            self.proteotypicity = None

    def prepare_output(self, charges_array, sequences_lengths):
        # prepare spectrum
        if self.spectrum is not None:
            self.spectrum.prepare_spectrum(charges_array=charges_array,
                                           sequences_lengths=sequences_lengths)

        # prepare irt
        if self.irt is not None:
            self.irt.prepare_irt()

        # prepare proteotypicity
        # nothing to prepare proteotypicity is only reported raw

    def assemble_dictionary(self):
        return_dictionary = {}


        if self.proteotypicity is not None:
            return_dictionary["proteotypicity"] = self.proteotypicity.raw

        if self.irt is not None:
            return_dictionary["irt"] = self.irt.normalized

        if self.spectrum is not None:
            return_dictionary["intensity"] = self.spectrum.intensity.filtered
            return_dictionary["fragmentmz"] = self.spectrum.mz.filtered
            return_dictionary["annotation_number"] = self.spectrum.annotation.filtered_number
            return_dictionary["annotation_type"] = self.spectrum.annotation.filtered_type
            return_dictionary["annotation_charge"] = self.spectrum.annotation.filtered_charge

        return return_dictionary


class PROSITproteotypicity:
    def __init__(self, pred_proteotyp):
        self.raw = pred_proteotyp


class PROSITirt:
    def __init__(self, pred_irt):
        self.raw = pred_irt
        self.normalized = None

    def normalize(self):
        self.normalized = [i * 43.39373 + 56.35363441 for i in self.raw]
        self.normalized = np.array(self.normalized)
        self.normalized.shape = self.raw.shape

    def prepare_irt(self):
        if type(self.raw) == np.ndarray:
            self.normalize()

class PROSITspectrum:
    def __init__(self, pred_intensity, sequences_array, charges_array):
        self.intensity = PROSITintensity(pred_intensity)

        fragment_mz_raw = np.array(
            [U.compute_ion_masses(sequences_array[i], charges_array[i])
            for i in range(len(sequences_array))])


        self.mz = PROSITfragmentmz(fragment_mz_raw)

        annot_raw_charge = np.array([C.ANNOTATION[1] for _ in range(len(sequences_array))])
        annot_raw_number = np.array([C.ANNOTATION[2] for _ in range(len(sequences_array))])
        annot_raw_type = np.array([C.ANNOTATION[0] for _ in range(len(sequences_array))])

        shape = (len(sequences_array), C.VEC_LENGTH)

        annot_raw_charge.shape = shape
        annot_raw_number.shape = shape
        annot_raw_type.shape = shape

        self.annotation = PROSITfragmentannotation(
            raw_charge=annot_raw_charge, 
            raw_number=annot_raw_number, 
            raw_type=annot_raw_type)

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
    def __init__(self, pred_intensity):
        self.raw = pred_intensity
        self.masked = None
        self.normalized = None
        self.filtered = None

    def normalize(self):
        self.normalized = U.normalize_intensities(self.masked)
        self.normalized[self.normalized < 0] = 0
        self.normalized[self.masked == -1] = -1


class PROSITfragmentmz:
    def __init__(self, fragment_mz_raw):
        self.raw = fragment_mz_raw
        self.masked = None
        self.filtered = None


class PROSITfragmentannotation:
    def __init__(self, raw_charge, raw_number, raw_type):
        self.raw_charge = raw_charge
        self.raw_number = raw_number
        self.raw_type = raw_type

        self.masked_charge = None
        self.masked_number = None
        self.masked_type = None

        self.filtered_charge = None
        self.filtered_number = None
        self.filtered_type = None
