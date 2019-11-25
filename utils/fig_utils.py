import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import itertools
import os


def profileplot(data, drops, data_type='shannon', out=None):
    """
    plot the profile color coded by p-value
    :param data: data frame with entropies\ deltag
    :param stats: corresponding p values data frame
    :param col: the sequence on which the plot should be generated
    :param data_type: the type of data used (shannon\ joint\ deltaG)
    :param out: optional - output file to save the plot
    :return: plot the graph
    """
    with sns.plotting_context(rc={"font.size":14,"axes.titlesize":18,"axes.labelsize":18,
                              "xtick.labelsize":14,"ytick.labelsize":14,'y.labelsize':16}):

        lower = 0.05

        #TODO generate stats + data together as in the jupyter notebook
        df = data

        drops = drops[drops['type'] == data_type]
        drops['list'] = drops.apply(lambda row: list(range(row['start'], row['end'] + 1)), axis=1)
        merged = list(itertools.chain(*drops['list'].values))

        df['stats'] = df['position'].apply(lambda x : 0 if x in merged else 1)

        slower = np.ma.masked_where(df['stats'].values > lower, df['value'].values)
        supper = np.ma.masked_where(df['stats'].values < lower, df['value'].values)

        plt.plot(df['position'], slower, color='#32B7DE', label = 'Drop')
        plt.plot(df['position'], supper, color='#DE3232', label='Non-drop')

        sns.despine()
        plt.legend()
        plt.xlabel('Position')
        plt.ylabel('value')
        plt.title('{} Profile'.format(data_type))

        if out != None:
            plt.savefig(os.path.join(out, '{}_profile.pdf'.format(data_type)), format='pdf', bbox_inches='tight', dpi=400)

def dropplot(data, seq, out = None):

    with sns.plotting_context(
            rc={"font.size": 12, "axes.labelsize": 15, "xtick.labelsize": 14, "ytick.labelsize": 12, 'aspect': 10}):
        f, ax = plt.subplots(figsize=(16, 4))

        shannon_drops = data[data['type'] == 'shannon']
        joint_drops = data[data['type'] == 'joint']

        pal = sns.color_palette("bwr")
        ax.plot([1, len(seq)], [1,1], color="black", alpha=0.7, linewidth=4)   # [1,1] is the location on the y axis of
                                                                            # the genome line
        for row in shannon_drops.iterrows():
            row = row[1]
            ax.scatter([row['start'], row['end']], [1, 1], marker='s', s=row['size'],
                       c=pal[int(row['rank'])])

        ax.plot([1, len(seq)], [2,2], color="black", alpha=0.7, linewidth=4)   # [1,1] is the location on the y axis of
                                                                            # the genome line
        for row in joint_drops.iterrows():
            row = row[1]
            ax.scatter([row['start'], row['end']], [2, 2], marker='s', s=row['size'],
                       c=pal[int(row['rank'])])


        ax.set_xlabel("Position in the genome")
        ax.get_yaxis().set_visible(False)
        if out != None:
            plt.savefig(os.path.join(out, 'genome_drops.pdf'), format='pdf', bbox_inches='tight', dpi=400)
