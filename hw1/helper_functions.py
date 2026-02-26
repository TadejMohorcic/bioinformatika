from typing import Tuple, Generator, List

from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord


def codons(seq: str) -> Generator[str, None, None]:
    """Walk along the string, three nucleotides at a time. Cut off excess."""
    for i in range(0, len(seq) - 2, 3):
        yield seq[i:i + 3]


def extract_gt_orfs(record, start_codons, stop_codons, validate_cds=True, verbose=False):
    """Extract the ground truth ORFs as indicated by the NCBI annotator in the
    gene coding regions (CDS regins) of the genome.

    Parameters
    ----------
    record: SeqRecord
    start_codons: List[str]
    stop_codons: List[str]
    validate_cds: bool
        Filter out NCBI provided ORFs that do not fit our ORF criteria.
    verbose: bool

    Returns
    -------
    List[Tuple[int, int, int]]
        tuples of form (strand, start_loc, stop_loc). Strand should be either 1
        for reference strand and -1 for reverse complement.

    """
    cds_regions = [f for f in record.features if f.type == "CDS"]

    orfs = []
    for region in cds_regions:
        loc = region.location
        seq = record.seq[loc.start.position:loc.end.position]
        if region.strand == -1:
            seq = seq.reverse_complement()
            
        if not validate_cds:
            orfs.append((region.strand, loc.start.position, loc.end.position))
            continue

        try:
            assert seq[:3] in start_codons, "Start codon not found!"
            assert seq[-3:] in stop_codons, "Stop codon not found!"
            # Make sure there are no stop codons in the middle of the sequence
            for codon in codons(seq[3:-3]):
                assert (
                    codon not in stop_codons
                ), f"Stop codon {codon} found in the middle of the sequence!"

            # The CDS looks fine, add it to the ORFs
            orfs.append((region.strand, loc.start.position, loc.end.position))

        except AssertionError as ex:
            if verbose:
                print(
                    "Skipped CDS at region [%d - %d] on strand %d"
                    % (loc.start.position, loc.end.position, region.strand)
                )
                print("\t", str(ex))
                
    # Some ORFs in paramecium have lenghts not divisible by 3. Remove these
    orfs = [orf for orf in orfs if (orf[2] - orf[1]) % 3 == 0]

    return orfs


def find_orfs(sequence, start_codons, stop_codons):
    """Find possible ORF candidates in a single reading frame.

    Parameters
    ----------
    sequence: Seq
    start_codons: List[str]
    stop_codons: List[str]

    Returns
    -------
    List[Tuple[int, int]]
        tuples of form (start_loc, stop_loc)

    """
    orf_candidates = []
    start_index = -1

    i = 0
    for codon in codons(sequence):
        if codon in start_codons:
            if start_index < 0:
                start_index = i

        i += 3
        
        if codon in stop_codons:
            if start_index >= 0:
                orf_candidates.append((start_index, i))
                start_index = -1
    
    return orf_candidates


def find_all_orfs(sequence, start_codons, stop_codons):
    """Find ALL the possible ORF candidates in the sequence using all six
    reading frames.

    Parameters
    ----------
    sequence: Seq
    start_codons: List[str]
    stop_codons: List[str]

    Returns
    -------
    List[Tuple[int, int, int]]
        tuples of form (strand, start_loc, stop_loc). Strand should be either 1
        for reference strand and -1 for reverse complement.

    """

    n = len(sequence)
    orfs = []

    for i in range(3):
        forward_orfs = find_orfs(sequence[i:], start_codons, stop_codons)
        backward_orfs = find_orfs(sequence.reverse_complement()[i:], start_codons, stop_codons)

        for start, stop in forward_orfs:
            orfs.append((1, start + i, stop + i))

        for start, stop in backward_orfs:
            orfs.append((-1, n - i - stop, n - i - start))

    return orfs


def translate_to_protein(seq):
    """Translate a nucleotide sequence into a protein sequence.

    Parameters
    ----------
    seq: str

    Returns
    -------
    str
        The translated protein sequence.

    """
    translate_dictionary = {'A':{'GCD','GCC','GCA','GCG'},'C':{'TGT','TGC'},'D':{'GAT','GAC'},'E':{'GAA','GAG'},'F':{'TTT','TTC'},'G':{'GGT','GGC','GGA','GGG'},'H':{'CAT','CAC'},'I':{'ATT','ATC','ATA'},'K':{'AAA','AAG'},'L':{'TTA','TTG','CTT','CTC','CTA','CTG'},'M':{'ATG'},'N':{'AAT','AAC'},'P':{'CCT','CCC','CCA','CCG'},'Q':{'CAA','CAG'},'R':{'CGT','CGC','CGA','CGG','AGA','AGG'},'S':{'TCT','TCC','TCA','TCG','AGT','AGC'},'T':{'ACT','ACC','ACA','ACG'},'V':{'GTT','GTC','GTA','GTG'},'W':{'TGG'},'Y':{'TAT','TAC'}}
    protein = ''
    for codon in codons(seq):
        if codon == 'ATG' and protein == '':
            protein += '.'
            continue

        if codon in {'TAA','TGA','TAG'}:
            return protein[1:]

        for key in translate_dictionary:
            if codon in translate_dictionary[key]:
                protein += key
                break


def find_all_orfs_nested(sequence, start_codons, stop_codons):
    """Bonus problem: Find ALL the possible ORF candidates in the sequence using
    the updated definition of ORFs.

    Parameters
    ----------
    sequence: Seq
    start_codons: List[str]
    stop_codons: List[str]

    Returns
    -------
    List[Tuple[int, int, int]]
        tuples of form (strand, start_loc, stop_loc). Strand should be either 1
        for reference strand and -1 for reverse complement.

    """
    n = len(sequence)
    orfs = []

    for i in range(3):
        forward_orfs = find_orfs_any(sequence[i:], start_codons, stop_codons)
        backward_orfs = find_orfs_any(sequence.reverse_complement()[i:], start_codons, stop_codons)

        for start, stop in forward_orfs:
            orfs.append((1, start + i, stop + i))

        for start, stop in backward_orfs:
            orfs.append((-1, n - i - stop, n - i - start))

    return orfs

def find_orfs_any(sequence, start_codons, stop_codons):
    """Find possible ORF candidates in a single reading frame.

    Parameters
    ----------
    sequence: Seq
    start_codons: List[str]
    stop_codons: List[str]

    Returns
    -------
    List[Tuple[int, int]]
        tuples of form (start_loc, stop_loc)

    """
    orf_candidates = []
    start_index = []

    i = 0
    for codon in codons(sequence):
        if codon in start_codons:
            start_index.append(i)

        i += 3
        
        if codon in stop_codons:
            for index in start_index:
                orf_candidates.append((index, i))
            start_index = []
    
    return orf_candidates