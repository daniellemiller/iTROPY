import numpy as np
import pandas as pd
from tqdm import tqdm
import statsmodels.stats.multitest as multi
import os
import glob, re


def stretchFinder(profile, l, m=10**4):
    """
    implementation of strechFinder as described in : "Synonymous site conservation in the HIV-1 genome"
    :param profile: a vector of entropy values
    :param l: the window size
    :param m: number of permutations
    :return:
    """
    start_index = []
    p_values = []

    #create a per-profile distribution of averages, then sample
    avgs = np.array([])
    for j in range(m):
        new_profile = profile
        cur_avg = np.mean(new_profile[np.random.choice(len(new_profile), size=l, replace=False)])
        avgs = np.insert(avgs, avgs.searchsorted(cur_avg), cur_avg)

    for i in tqdm(range(0,len(profile) - l)):
        # get the current window and its average value
        w = profile[i:i+l]
        avg = np.mean(w)

        # sort average in order to get the p value
        idx = np.searchsorted(avgs, avg)
        p_value = idx/m
        p_values.append(p_value)
        start_index.append(i)

    data =  pd.DataFrame({'start':start_index, 'p_value':p_values, 'l':l})

    # correct for multiple tests
    data['corrected_pvalue'] = multi.fdrcorrection(data['p_value'])[1]

    return data

def find_sequential_drops(shannon, joint, alias=None, output=None, t=0.05):
    """
    given a family generate a table of all sequential drops
    :param shannon: shannon entropy matrix
    :param joint: joint entropy matrix
    :param t: p value threshold
    :return: data frame with stat and end points of every significant drop
    """

    col_dfs = []

    # for each feature in the matrix calculate the number of drops. locations and sizes

    x = shannon["p_value"].dropna()
    y = joint["p_value"].dropna()

    # get only significant drops
    x = x[x <= t]
    y = y[y <= t]

    # remember index as a new column - had bugs with splitting by index later...
    x = x.reset_index()
    y = y.reset_index()

    dx = np.diff(x['index'])
    dy = np.diff(y['index'])

    x_pos = [i + 1 for i in np.where(dx > 1)[0]]
    y_pos = [i + 1 for i in np.where(dy > 1)[0]]

    x_mod = [0] + x_pos + [len(x) + 1]
    y_mod = [0] + y_pos + [len(y) + 1]

    x_splits = [x.iloc[x_mod[n]:x_mod[n + 1]] for n in range(len(x_mod) - 1)]
    y_splits = [y.iloc[y_mod[n]:y_mod[n + 1]] for n in range(len(y_mod) - 1)]
    # fil start and end positions creating a dataframe for each column
    x_starts = []
    x_ends = []

    for s in x_splits:
        x_starts.append(s['index'].iloc[0])
        x_ends.append(s['index'].iloc[-1])

    y_starts = []
    y_ends = []

    for s in y_splits:
        y_starts.append(s['index'].iloc[0])
        y_ends.append(s['index'].iloc[-1])

    shannon_drops = pd.DataFrame({'start': x_starts, 'end': x_ends, 'type': 'shannon', 'seq': alias})
    joint_drops = pd.DataFrame({'start': y_starts, 'end': y_ends, 'type': 'joint', 'seq': alias})

    col_dfs.append(shannon_drops)
    col_dfs.append(joint_drops)

    drops = pd.concat(col_dfs)
    drops['size'] = drops['end'] - drops['start'] + 1

    if output != None:
        drops.to_csv(output, index=False)

    return drops

def merge_stats_by_type(folder, data_type):
    """
    the method merges all stats csv files into a single file
    :param family: the family name
    :param joint: indicator for joint entropy. if true run the analysis on joint entropy
    :return: saves a csv with all p values
    """

    # define the folder from which all stats are going to be generated
    stats_files = glob.glob(os.path.join(folder, '*{}*seq*stats.csv'.format(data_type)))
    mapping = {}
    for f in stats_files:
        df = pd.read_csv(f)
        seq_id = re.findall(r'seq_\d+', f)[0]
        mapping[seq_id] = df['corrected_pvalue']

    # create a complete dataframe with all p values
    stats_by_seq = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in mapping.items()]))

    output = os.path.join(folder, '{}_all_stats.csv'.format(data_type))

    stats_by_seq.to_csv(output, index=False)
    return stats_by_seq

def merge_all_drops(shannon, joint, drops):
    """
    merge all information on drops
    :param shannon: data frame with shannon entropy profiles by seq
    :param joint: data frame with joint entropy profiles by seq
    :param drops: the output of find sequential drops
    :return:
    """
    all_sequences = list(shannon.columns)
    dfs = []
    for seq in tqdm(all_sequences):
        seq_drops = drops[drops['seq'] == seq]
        seq_drops['position'] = seq_drops.apply(lambda row: list(range(row['start'], row['end'] + 1)), axis=1)

        # melt
        seq_drops = pd.DataFrame([[a, b, c, d, e, p] for a, b, c, d, e, P in \
                                  seq_drops.values for p in P], columns=seq_drops.columns)

        # merge with sequence values by position

        cur_shannon = shannon[seq].reset_index()
        cur_shannon['position'] = cur_shannon['index'] + 1
        cur_shannon['shannon'] = cur_shannon[seq]

        cur_joint = joint[seq].reset_index()
        cur_joint['position'] = cur_joint['index'] + 1
        cur_joint['joint'] = cur_joint[seq]

        cur_shannon.drop(columns=['index', seq], inplace=True)
        cur_joint.drop(columns=['index', seq], inplace=True)

        merged = pd.merge(seq_drops, cur_shannon, on=['position'])
        merged = pd.merge(merged, cur_joint, on=['position'])
        dfs.append(merged)

    result = pd.concat(dfs)
    return result