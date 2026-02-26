from collections import defaultdict

def global_alignment(seq1, seq2, scoring_function):
    """Global sequence alignment using the Needleman–Wunsch algorithm.

    Indels should be denoted with the "-" character.

    Parameters
    ----------
    seq1: str
        First sequence to be aligned.
    seq2: str
        Second sequence to be aligned.
    scoring_function: Callable

    Returns
    -------
    str
        First aligned sequence.
    str
        Second aligned sequence.
    float
        Final score of the alignment.

    Examples
    --------
    >>> global_alignment("abracadabra", "dabarakadara", lambda x, y: [-1, 1][x == y])
    ('-ab-racadabra', 'dabarakada-ra', 5.0)

    Other alignments are not possible.

    """
    score_matrix = defaultdict(int)
    backtracking = {}

    score_matrix[0, 0] = 0

    for i in range(1, len(seq1) + 1):
        score_matrix[i, 0], backtracking[i, 0] = float(score_matrix[i - 1, 0] + scoring_function('*', seq1[i - 1])), (i - 1, 0)

    for i in range(1, len(seq2) + 1):
        score_matrix[0, i], backtracking[0, i] = float(score_matrix[0, i - 1] + scoring_function('*', seq2[i - 1])), (0, i - 1)

    for i in range(1, len(seq1) + 1):
        for j in range(1, len(seq2) + 1):
            score_matrix[i, j], backtracking[i, j] = max((float(score_matrix[i - 1, j] + scoring_function('*', seq1[i - 1])), (i - 1, j)),
                                                         (float(score_matrix[i, j - 1] + scoring_function('*', seq2[j - 1])), (i, j - 1)),
                                                         (float(score_matrix[i - 1, j - 1] + scoring_function(seq1[i - 1], seq2[j - 1])), (i - 1, j - 1)))

    final_score = score_matrix[max(score_matrix)]

    alignment_1, alignment_2 = '', ''
    current = (len(seq1), len(seq2))
    
    while current != (0, 0):
        prev = backtracking[current]
        alignment_1 += '-' if prev[0] == current[0] else seq1[prev[0]]
        alignment_2 += '-' if prev[1] == current[1] else seq2[prev[1]]
        current = backtracking[current]

    return alignment_1[::-1], alignment_2[::-1], final_score


def local_alignment(seq1, seq2, scoring_function):
    """Local sequence alignment using the Smith-Waterman algorithm.

    Indels should be denoted with the "-" character.

    Parameters
    ----------
    seq1: str
        First sequence to be aligned.
    seq2: str
        Second sequence to be aligned.
    scoring_function: Callable

    Returns
    -------
    str
        First aligned sequence.
    str
        Second aligned sequence.
    float
        Final score of the alignment.

    Examples
    --------
    >>> local_alignment("pending itch", "unending glitch", lambda x, y: [-1, 1][x == y])
    ('ending --itch', 'ending glitch', 9.0)

    Other alignments are not possible.

    """
    score_matrix = defaultdict(int)
    backtracking = {}

    for i in range(len(seq1) + 1):
        score_matrix[i, 0] = 0

    for i in range(len(seq2) + 1):
        score_matrix[0, i] = 0

    for i in range(1, len(seq1) + 1):
        for j in range(1, len(seq2) + 1):
            score_matrix[i, j], backtracking[i, j] = max((float(score_matrix[i - 1, j - 1] + scoring_function(seq1[i - 1], seq2[j - 1])), (i - 1, j - 1)),
                                                         (float(score_matrix[i, j - 1] + scoring_function('*', seq2[j - 1])), (i, j - 1)),
                                                         (float(score_matrix[i - 1, j] + scoring_function('*', seq1[i - 1])), (i - 1, j)),
                                                         (0, (0, 0)))

    best_alignment = max(score_matrix.values())
    endings = [k for k, v in score_matrix.items() if v == best_alignment] # find all max value ends, return only one of them
    end = endings[0]
    alignment_1, alignment_2 = '', ''

    while score_matrix[end] != 0:
        prev = backtracking[end]
        alignment_1 += '-' if prev[0] == end[0] else seq1[prev[0]]
        alignment_2 += '-' if prev[1] == end[1] else seq2[prev[1]]
        end = prev

    return alignment_1[::-1], alignment_2[::-1], best_alignment
