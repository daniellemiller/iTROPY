from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
import matplotlib.pyplot as plt
import seaborn as sns

from utils.utils import *
from utils.stats_utils import stretchFinder
import pickle

class Drops:
    def __init__(self, profiler, k=5):
        self.profiler = profiler
        self.k = k
        self.shannon = None
        self.joint = None
        self.drops=None
        self.X = None
        self.inference_cols = ['Shannon_k{}'.format(k) for k in [1, 2, 3, 4, 5]] + ['Joint_k{}'.format(k) for k in
                                                                                [1, 2, 3, 4, 5]] + ['DeltaG',
                                                                                                    'position']

    def get_drops(self):
        """
        generate all shannon and joint entropy drops
        """
        shannon_drops = stretchFinder(profile=self.profiler.matrix[f'Shannon_k{self.k}'].values, l=self.profiler.l)
        joint_drops = stretchFinder(profile=self.profiler.matrix[f'Joint_k{self.k}'].values, l=self.profiler.l)

        all_drops = find_sequential_drops(shannon_drops, joint_drops, output=None)
        self.shannon = all_drops[all_drops['type'] == 'shannon']
        self.joint = all_drops[all_drops['type'] == 'joint']
        self.drops = find_drops_overlaps(self.shannon, self.joint)
        self.drops['drop_id'] = self.drops.apply(lambda row: "Drop {}".format(row.name), axis=1)

    def GMM_fit(self, k=4):
        """
        fits GMM model to a given matrix
        :param k: number of clusters. default 4
        :return: a data frame containing cluster assignments
        """
        cols_2_consider = self.inference_cols
        gmm = GaussianMixture(n_components=k)
        gmm.fit(self.profiler.matrix[cols_2_consider])
        clusters_gmm = gmm.predict(self.profiler.matrix[cols_2_consider])
        proba = gmm.predict_proba(self.profiler.matrix[cols_2_consider])
        self.profiler.matrix['GMM_clusters'] = clusters_gmm

        # add the label as a string
        self.profiler.matrix['GMM_clusters'] = self.profiler.matrix['GMM_clusters'].apply(lambda x: str(int(x)))
        # add probabilities of each point to each cluster
        for i in range(k):
            self.profiler.matrix['prob_cluster_{}'.format(i)] = proba[:, i]
        return self.profiler.matrix

    def update_profiler(self):
        """
        update the profiler matrix to contain all information on drops
        """
        self.profiler.matrix['drop_id'] = self.profiler.matrix['position'].apply(lambda x: get_drop_id_by_pos(x, self.drops))
        self.profiler.matrix= self.profiler.matrix[self.profiler.matrix['drop_id'] != 'no drop']
        self.GMM_fit()

        # add ranks to each drop
        score_2_cluster = []
        for cluster in ["0", "1", "2", "3"]:
            score_2_cluster.append(tuple((cluster, cluster_score(
                self.profiler.matrix[self.profiler.matrix['GMM_clusters'] == cluster]))))

        ranking = score_to_rank(score_2_cluster)
        self.profiler.matrix['rank'] = self.profiler.matrix['GMM_clusters'].apply(lambda x: ranking[x])

        for c in tqdm(self.inference_cols):
            self.profiler.matrix[c] = self.profiler.matrix[c].astype(float)

        self.profiler.matrix['GMM_clusters'] = self.profiler.matrix['GMM_clusters'].astype(int)     # change to int for agg
        data = self.profiler.matrix.groupby(['drop_id', 'seq_id', 'family']).agg({np.mean, np.median, np.std}).reset_index()
        data.columns = [' '.join(col).strip() for col in data.columns.values] # flatten columns names
        #hard coded columns for testing the model
        cols = ['DeltaG mean', 'DeltaG median', 'DeltaG std', 'GMM_clusters mean',
                'GMM_clusters median', 'GMM_clusters std', 'Joint_k1 mean',
                'Joint_k1 median', 'Joint_k1 std', 'Joint_k2 mean', 'Joint_k2 median',
                'Joint_k2 std', 'Joint_k3 mean', 'Joint_k3 median', 'Joint_k3 std',
                'Joint_k4 mean', 'Joint_k4 median', 'Joint_k4 std', 'Joint_k5 mean',
                'Joint_k5 median', 'Joint_k5 std', 'Shannon_k1 mean',
                'Shannon_k1 median', 'Shannon_k1 std', 'Shannon_k2 mean',
                'Shannon_k2 median', 'Shannon_k2 std', 'Shannon_k3 mean',
                'Shannon_k3 median', 'Shannon_k3 std', 'Shannon_k4 mean',
                'Shannon_k4 median', 'Shannon_k4 std', 'Shannon_k5 mean',
                'Shannon_k5 median', 'Shannon_k5 std', 'drop_id', 'family',
                'position mean', 'seq_id']
        data = data[cols]

        # rename columns
        data = data.rename(columns={'position mean': 'position'})
        data['DeltaG median'] = data['DeltaG median'].apply(lambda x: 0 if x == -0.0 else x)
        data['scaled_DeltaG'] = data['DeltaG median'].apply(lambda x: rescale(x))
        data = pd.merge(data, self.drops[['drop_id', 'start', 'end', 'size']], on=['drop_id'])
        data['nucleotide_info'] = data.apply(lambda row: extract_nucleotide_information(
            self.profiler.seq, row['start'], row['end']), axis=1)

        data['A_rel_genome'] = data['nucleotide_info'].apply(lambda lst: float(lst[0][0]))
        data['A_rel_drop'] = data['nucleotide_info'].apply(lambda lst: float(lst[0][-1]))
        data['C_rel_genome'] = data['nucleotide_info'].apply(lambda lst: float(lst[1][0]))
        data['C_rel_drop'] = data['nucleotide_info'].apply(lambda lst: float(lst[1][-1]))
        data['G_rel_genome'] = data['nucleotide_info'].apply(lambda lst: float(lst[2][0]))
        data['G_rel_drop'] = data['nucleotide_info'].apply(lambda lst: float(lst[2][-1]))
        data['T_rel_genome'] = data['nucleotide_info'].apply(lambda lst: float(lst[3][0]))
        data['T_rel_drop'] = data['nucleotide_info'].apply(lambda lst: float(lst[3][-1]))

        data = data.rename(columns={'size': 'drops_size'})
        self.X = data
        return self.X

    def categorize_drops(self):
        self.drops['Category'] = self.drops.apply(lambda row: _map_drops(row['drop_id'], row['type'], self.X), axis=1)
        return

    def rank_drops(self):
        """
        rank drops by previously trained GBR model
        """
        self.X['SL'] = len(self.profiler.seq)
        self.X['SL_class'] = 1
        self.X['ds_class'] = 1
        with open(r'../data/trained_GBR.pickle', 'rb') as f:
            mdl = pickle.load(f)
        pipeline = _make_preprocessing_pipeline()
        X_test = pipeline.fit_transform(self.X.drop(columns=['nucleotide_info']))
        predicted = mdl.predict(X_test)
        #get ranking and not predicted value (https://stackoverflow.com/questions/5284646/rank-items-in-an-array-using-python-numpy-without-sorting-array-twice/)
        temp = predicted.argsort()
        ranks = np.empty_like(temp)
        ranks[temp] = np.arange(len(predicted))
        self.X['rank'] = ranks

        self.drops = pd.merge(self.drops, self.X[['drop_id', 'rank']], on=['drop_id']).sort_values(by=['rank'])


    def dropplot(self, feature='rank'):
        mapping = {}
        if feature not in self.X.columns:
            raise Exception(f"Not a valid coloring feature, should be one of {self.X.columns}")
        vals = np.sort(self.X[feature].unique())
        genome_len = len(self.profiler.seq)

        for i, cons in enumerate(vals):
            mapping[str(cons)] = i

        n_colors = 2
        if vals.shape[0] > 2:
            n_colors = max(8, vals.shape[0])

        with sns.plotting_context(
                rc={"font.size": 12, "axes.labelsize": 15, "xtick.labelsize": 14, "ytick.labelsize": 12, 'aspect': 10}):
            pal = sns.mpl_palette('seismic', n_colors)
            f, ax = plt.subplots(figsize=(14, 4))
            X = self.X
            ax.plot([1, genome_len], [0, 0], color="black", alpha=0.7, linewidth=4)
            for i, row in X.iterrows():
                ax.scatter([row['start'], row['end']], [0, 0], marker='s', s=2 * row['size'],
                           c=pal[mapping[str(row[feature])]])
            #TODO replca with a proper color bar besides the plot
            sns.palplot(sns.mpl_palette('seismic', n_colors))
            plt.show()


def _make_preprocessing_pipeline():
    encoder = Pipeline([
        ("encoding", ColumnTransformer([
            ("family", OrdinalEncoder(), ["family"]),
            ("drop_id", OrdinalEncoder(), ["drop_id"]),
            ("seq_id", OrdinalEncoder(), ["seq_id"]),
        ], remainder="passthrough"))
    ])
    return encoder

def _map_drops(drop_id, drop_type, data):
    value = data[data['drop_id'] == drop_id]['DeltaG median'].values[0]
    if drop_type == 'joint':
        return 'Structural'
    elif value <= 0.4:
        return 'Repetitive'
    else:
        return 'Structural'

