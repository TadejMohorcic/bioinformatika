import numpy as np

def jukes_cantor(reference_sequence: str, distant_sequence: str) -> float:
    """The Jukes-Cantor correction for estimating genetic distances
    calculated with Hamming distance.
    Should return genetic distance with the same unit as if not corrected.

    Parameters
    ----------
    reference_sequence: str
        A string of nucleotides in a sequence used as a reference
        in an alignment with other (e.g. AGGT-GA)
    distant_sequence: str
        A string of nucleotides in a sequence after the alignment
        with a reference (e.g. AGC-AGA)

    Returns
    -------
    float
        The Jukes-Cantor corrected genetic distance using Hamming distance.
        For example 1.163.

    """

    length, difference = 0, 0

    # Go over all nucleotides and count the different ones.
    for i in range(len(reference_sequence)):
        if reference_sequence[i] == '-' or distant_sequence[i] == '-':
            continue
        else:
            if reference_sequence[i] != distant_sequence[i]:
                difference += 1

            length += 1

    # Calculate the corrected difference.
    p = difference / length
    distance = -3/4 * np.log(1 - 4/3 * p)
    corrected_distance = length * distance
    
    return corrected_distance


def kimura_two_parameter(reference_sequence: str, distant_sequence: str) -> float:
    """The Kimura Two Parameter correction for estimating genetic distances
    calculated with Hamming distance.
    Should return genetic distance with the same unit as if not corrected.

    Parameters
    ----------
    reference_sequence: str
        A string of nucleotides in a sequence used as a reference
        in an alignment with other (e.g. AGGT-GA)
    distant_sequence: str
        A string of nucleotides in a sequence after the alignment
        with a reference (e.g. AGC-AGA)

    Returns
    -------
    float
        The Kimura corrected genetic distance using Hamming distance.
        For example 1.196.

    """
    length, transitional_diff, transversional_diff = 0, 0, 0

    # Transitionons are A <-> G and C <-> T, and transversions are A <-> (C, T) and G <-> (C, T).
    transitions = {('A','G'),('C','T')}
    transversions = {('A','C'),('A','T'),('G','C'),('G','T')}

    # Go over all nucleotides and count the different ones.
    for i in range(len(reference_sequence)):
        if reference_sequence[i] == '-' or distant_sequence[i] == '-':
            continue
        else:
            nuc_1, nuc_2 = reference_sequence[i], distant_sequence[i]

            if (nuc_1, nuc_2) in transitions or (nuc_2, nuc_1) in transitions:
                transitional_diff += 1
            elif (nuc_1, nuc_2) in transversions or (nuc_2, nuc_1) in transversions:
                transversional_diff += 1

            length += 1

    # Calculate the corrected difference.
    p, q = transitional_diff / length, transversional_diff / length
    distance = (-1/2 * np.log((1 - 2 * p - q) * np.sqrt(1 - 2 * q)))
    corrected_distance = length * distance
    
    return corrected_distance