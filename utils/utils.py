import math
from Bio.Seq import Seq
import random
import RNA
from functools import reduce
from sklearn.mixture import GaussianMixture
from utils.stats_utils import *
from keras.models import Model
from keras.layers import Dense, Input

# k-mers distribution based entropy measures.
def entropy_by_kmer(seq, k):
    """
    calculate the entropy of a sequence according to its k-mers
    :param seq: a genome string
    :param k: size of the k-mer to use
    :return: entropy
    """

    # update kmers
    kmers = {}
    for i in range(len(seq) - k):
        kmer = seq[i:i+k]
        if kmer in kmers:
            kmers[kmer] += 1
        else:
            kmers[kmer] = 1

    # calculate entropy
    total_kmers = sum(kmers.values())
    entropy = 0
    for kmer in kmers:
        p = kmers[kmer] / total_kmers
        entropy += -(p * math.log2(p))

    return entropy

def joint_entropy (seq1, seq2, k):
    """
    calculates the joint entropy of two sequences.
    :param seq1: sequence #1
    :param seq2: sequence #2
    :param k: k-mer length
    :return: joint entropy value
    """

    kmers_1 = {}
    kmers_2 = {}

    # kmers in sequence #1
    for i in range(len(seq1) - k):
        kmer = seq1[i:i+k]
        if kmer in kmers_1:
            kmers_1[kmer] += 1
        else:
            kmers_1[kmer] = 1

    for i in range(len(seq2) - k):
        kmer = seq2[i:i+k]
        if kmer in kmers_2:
            kmers_2[kmer] += 1
        else:
            kmers_2[kmer] = 1

    # calculate joint entropy
    total_kmers_1 = sum(kmers_1.values())
    total_kmers_2 = sum(kmers_2.values())

    total = total_kmers_1 + total_kmers_2

    # compare the kmers space to be equal at both
    for kmer in kmers_1:
        if kmer not in kmers_2:
            kmers_2[kmer] = 0

    for kmer in kmers_2:
        if kmer not in kmers_1:
            kmers_2[kmer] = 0

    joint_entropy = 0
    for kmer1 in kmers_1:
        for kmer2 in kmers_2:
            p_xy = (kmers_1[kmer1] + kmers_2[kmer2]) / total

            joint_entropy += -(p_xy * math.log2(p_xy))

    return joint_entropy


def information_storage (seq1, seq2, k):
    """
    calculates the information storage of two sequences.
    :param seq1: sequence #1
    :param seq2: sequence #2
    :param k: k-mer length
    :return: information storage value
    """

    kmers_1 = {}
    kmers_2 = {}

    # kmers in sequence #1
    for i in range(len(seq1) - k):
        kmer = seq1[i:i+k]
        if kmer in kmers_1:
            kmers_1[kmer] += 1
        else:
            kmers_1[kmer] = 1

    for i in range(len(seq2) - k):
        kmer = seq2[i:i+k]
        if kmer in kmers_2:
            kmers_2[kmer] += 1
        else:
            kmers_2[kmer] = 1

    # calculate joint entropy
    total_kmers_1 = sum(kmers_1.values())
    total_kmers_2 = sum(kmers_2.values())

    total = total_kmers_1 + total_kmers_2

    # compare the kmers space to be equal at both
    for kmer in kmers_1:
        if kmer not in kmers_2:
            kmers_2[kmer] = 0

    for kmer in kmers_2:
        if kmer not in kmers_1:
            kmers_2[kmer] = 0

    inf_storage = 0
    for kmer1 in kmers_1:
        for kmer2 in kmers_2:

            p_xy = (kmers_1[kmer1] + kmers_2[kmer2]) / total
            p_x = kmers_1[kmer1] / total_kmers_1
            p_y = kmers_2[kmer2] / total_kmers_2

            if p_x == 0 or p_y == 0:
                continue
            inf_storage += p_xy * math.log2(p_xy/(p_x*p_y))

    return inf_storage



def get_reverse_complement(seq):
    """
    get reverse complement genome
    :param seq: a genome sequence
    :return: a string of reverse complement
    """
    seq = Seq(seq)
    reverse_complement = seq.reverse_complement()
    return reverse_complement



# shuffle scrambler for a sequence.
def _scrambler(word):
    word_to_scramble = list(word)
    random.shuffle(word_to_scramble)
    new_word = ''.join(word_to_scramble)
    return new_word


def scrambler(word):
   new_word = _scrambler(word)
   while new_word == word and len(word) > 1:
       new_word = _scrambler(word)
   return new_word


### profiles generator ###
def get_joint_entropy_profile_per_sequence(seq, w, alias, k=5, out=None):
    """
    sliding window entropy profile of all sequences in a family
    :param fasta: a fasta file contatining viral sequences
    :param w: the window size
    :param out: optional. if != None a profile will be saved as a png
    :return: the vector of profile entropy
    """
    entropies = []
    # get identifier and genomic sequence
    genome = seq

    for j in range(len(genome) - w):
        sub_genome = genome[j:j+w]
        try:
            rc_sub_genome = str(get_reverse_complement(sub_genome))
            entropy = joint_entropy(sub_genome, rc_sub_genome, k)
            entropies.append(entropy)
        except:
            break

    df = pd.DataFrame({'{}'.format(alias):entropies})
    if out != None:
        df.to_csv(os.path.join(out, '{}_Joint_profile.csv'.format(alias)), index=False)

    return df


def get_entropy_profile_per_sequence(seq, w, alias, k=5, out=None):
    """
    sliding window entropy profile of all sequences in a family
    :param fasta: a fasta file contatining viral sequences
    :param w: the window size
    :param out: optional. if != None a profile will be saved as a png
    :return: the vector of profile entropy
    """

    entropies = []
    # get identifier and genomic sequence
    genome = seq

    for j in range(len(genome) - w):
        sub_genome = genome[j:j+w]
        entropy = entropy_by_kmer(sub_genome, k)
        entropies.append(entropy)


    df = pd.DataFrame({'{}'.format(alias):entropies})
    if out != None:
        df.to_csv(os.path.join(out, '{}_Shannon_profile.csv'.format(alias)), index=False)

    return df


def deltaG_calculator(seq):
    """
    calculate the minimum free energy (G) for a given sequence
    :param seq: an rna sequence
    :return: minimum free energy
    """
    ss, mfe = RNA.fold(seq)
    return mfe


def deltaG_profile_per_sequence(seq, w, alias, out=None):
    """
    sliding window free energy profile of all sequences in a family
    :param seq: a string containing a sequence
    :param w: the window size
    :param out: output file
    :return: the vector of profile entropy
    """
    genome = seq
    values = []

    for j in range(len(genome) - w):
        sub_genome = genome[j:j+w]
        mfe = deltaG_calculator(sub_genome)
        values.append(mfe)

    df = pd.DataFrame({'{}'.format(alias): values})
    if out != None:
        df.to_csv(os.path.join(out, '{}_deltaG_profile.csv'.format(alias)), index=False)

    return df


def generate_train_matrix(seq, w=200):
    """
    generate the input matrix to train in the GMM model. feature generation + normalization
    :param seq: a sequence of a genome
    :param w: widow size for entropy profile
    :return: data frame with all features
    """

    ks = [1, 2, 3, 4, 5]
    dfs = []
    for k in tqdm(ks):
        data_type = 'Shannon'
        alias = data_type + '_k' + str(k)
        profile = get_entropy_profile_per_sequence(seq=seq, w=w, alias=alias, k=k)
        profile = profile / profile.max()
        profile['position'] = profile.index + 1
        dfs.append(profile)

    for k in tqdm(ks):
        data_type = 'Joint'
        alias = data_type + '_k' + str(k)
        profile = get_joint_entropy_profile_per_sequence(seq=seq, w=w, alias=alias, k=k)
        profile = profile / profile.max()
        profile['position'] = profile.index + 1
        dfs.append(profile)

    # delta G profile is not dependent on k
    data_type = 'DeltaG'
    alias = data_type
    profile = deltaG_profile_per_sequence(seq=seq, w=w, alias=alias)
    profile = profile / profile.min()
    profile['position'] = profile.index + 1
    dfs.append(profile)

    mat = reduce(lambda left,right: pd.merge(left,right, on=['position']), dfs)
    return mat


def generate_list_of_sequetial_positions(drops):
    """
    create a list of positions [start_i, end_i for i in drops]
    :param drops: a data frame of a single sequence drops information
    :return: a list of all positions
    """
    l_start = drops['start'].values
    l_end = drops['end'].values

    l_combined = []
    for i in range(len(l_start)):
        l_combined.append(l_start[i])
        l_combined.append(l_end[i])
    return np.array(l_combined)


def get_drop_id_by_pos(pos, drops):
    """
    merge between drop position and
    :param pos:
    :param drops:
    :return:
    """

    all_drop_positions = generate_list_of_sequetial_positions(drops)
    pos_idx = all_drop_positions.searchsorted(pos)

    # check if its between drops or within drop
    set_start = -1
    set_end = -1

    if pos_idx % 2 != 0: # within drop
        set_start = all_drop_positions[pos_idx - 1]
        set_end = all_drop_positions[min(pos_idx, all_drop_positions.shape[0] - 1)]
    else:
        if pos in all_drop_positions:
            set_start = pos
            set_end = all_drop_positions[min(pos_idx + 1, all_drop_positions.shape[0] - 1)]

    if set_start == -1 or set_end == -1:
        return "no drop"
    else:
        val = drops[(drops['start'] == set_start) & (drops['end'] == set_end)]['drop_id'].values[0]
        return val

def _intersect(start,end, drops):
    """
    helper function for overlap detection
    """
    d = drops[drops['start'] < end]
    d['inter'] = d.apply(lambda row:len(list(set(range(start, end + 1)) & set(range(row['start'], row['end'] + 1)))), axis=1)
    #define pverlap as a minimun 1/3 cover
    if d[d['inter'] >= (end-start+1)//3].shape[0] != 0:
        return True
    return False


def find_drops_overlaps(shannon, joint):
    """
    return a single drop file, combining shannon and joint
    :param shannon: shannon drops
    :param joint: joint drops
    :return: a unite dataframe with -shhanon-joint overlaps only
    """
    s1 = shannon.copy()
    j1 = joint.copy()
    s1['isDrop'] = s1.apply(lambda row: _intersect(row['start'], row['end'], joint), axis=1)
    s1['type'] = 'shannon-joint'
    j1['isDrop'] = s1.apply(lambda row: _intersect(row['start'], row['end'], shannon), axis=1)
    return pd.concat([s1[s1['isDrop'] == True], j1[j1['isDrop'] == False]]).\
        reset_index(drop=True).drop(columns=['isDrop']).sort_values(by=['start','end'])


def rescale(value, max_range=0.6, min_range=0.3):
    """
    rescale deltaG values by random distribution
    :param value: actual value
    :param max_range: max range
    :param min_range: min range
    :return: rescaled value
    """
    scaled = value

    if value > min_range:
        scaled = (value) * (max_range - min_range) + min_range

    return scaled


def cluster_score(cluster):
    """
    calculate the score of a cluster by feature properties
    :param cluster: data frame containing the cluster matrix
    :return: a numeric value of the result
    """

    cols_2_consider = ['Shannon_k{}'.format(k) for k in [1, 2, 3, 4, 5]] + ['Joint_k{}'.format(k) for k in
                                                                            [1, 2, 3, 4, 5]] + ['DeltaG']
    total_score = 0
    for c in cols_2_consider:
        med = cluster[c].median()
        std = cluster[c].std()

        if c == 'DeltaG':
            total_score += (1 / (med*(1-med))) * std
        else:
            total_score += (1 / med ) * std
    return total_score

def score_to_rank(scores_mapping):
    """
    translate the scores into rankings
    :param scores_mapping: a list of tuples containing the score for each cluster
    :return: a mapping of cluster to rank
    """
    # sort the list by value in tup[1]
    sorted_scores = sorted(scores_mapping, key = lambda tup:tup[1], reverse=True)
    cluster_2_rank = {tup[0]:sorted_scores.index(tup)+1 for tup in sorted_scores}
    return cluster_2_rank


def extract_nucleotide_information(seq, start, end):
    """
    this function returns the relative nucleotide ratio comparing the the drop and to the entire genome
    :param seq: genomic sequence
    :param start: start position. int
    :param end: end position. int
    :return: a list with tuples for each nucleotide in the following order: A,C,G,T. (nuc relative in drop, nuc relative to genome)
    """
    drop_sequence = seq[start: end + 1]
    a_total = seq.count('a')
    c_total = seq.count('c')
    g_total = seq.count('g')
    t_total = seq.count('t')

    a_in_drop = drop_sequence.count('a')
    c_in_drop = drop_sequence.count('c')
    g_in_drop = drop_sequence.count('g')
    t_in_drop = drop_sequence.count('t')

    len_drop = len(drop_sequence)

    a_rel_genome = a_in_drop / a_total
    c_rel_genome = c_in_drop / c_total
    g_rel_genome = g_in_drop / g_total
    t_rel_genome = t_in_drop / t_total

    a_rel_drop = a_in_drop / len_drop
    c_rel_drop = c_in_drop / len_drop
    g_rel_drop = g_in_drop / len_drop
    t_rel_drop = t_in_drop / len_drop

    return [(a_rel_genome, a_rel_drop), (c_rel_genome, c_rel_drop), (g_rel_genome, g_rel_drop),
            (t_rel_genome, t_rel_drop)]