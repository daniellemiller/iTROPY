
from utils.utils import *


class Profiler:
    def __init__(self, seq, out_dir=None, w=200, l =100, alias='Seq', family='Unknown'):
        self.seq = seq
        self.w = w
        self.l = l
        self.out = out_dir
        self.matrix = None
        self.alias = alias
        self.family = family

        if not isinstance(self.seq, str):
            raise Exception(f"Seq data type not valid, expected string but got {type(self.seq)}")
        if not isinstance(self.w, np.number):
            raise Exception(f"Window size data type not valid, expected a number but got {type(self.w)}")
        if not isinstance(self.l, np.number):
            raise Exception(f"Permiutation window size data type not valid, expected number but got {type(self.l)}")
        if not isinstance(self.alias, str):
            raise Exception(f"Sequence alias data type not valid, expected str but got {type(self.alias)}")
        if self.w > len(self.seq):
            raise Exception(f"Provided window size {self.w} should be smaller than sequence length {len(self.seq)}")
        if self.l > len(self.seq):
            raise Exception(f"Provided permutation window size {self.l} should be smaller than sequence length {len(self.seq)}")

        if not os.path.exists(self.out):
            os.mkdir(self.out)

        self.seq = self.seq.lower()

    def get_profiles(self):
        """
        generate the input matrix to train in the GMM model. feature generation + normalization
        :param seq: a sequence of a genome
        :param seq_id: sequence identifier
        :param w: widow size for entropy profile
        :return: data frame with all features
        """
        ks = [1, 2, 3, 4, 5]
        dfs = []
        for k in tqdm(ks):
            data_type = 'Shannon'
            alias = data_type + '_k' + str(k)
            profile = get_entropy_profile_per_sequence(seq=self.seq, w=self.w, alias=alias, k=k)
            profile = profile / profile.max()
            profile['position'] = profile.index + 1
            dfs.append(profile)

        for k in tqdm(ks):
            data_type = 'Joint'
            alias = data_type + '_k' + str(k)
            profile = get_joint_entropy_profile_per_sequence(seq=self.seq, w=self.w, alias=alias, k=k)
            profile = profile / profile.max()
            profile['position'] = profile.index + 1
            dfs.append(profile)

        # delta G profile is not dependent on k
        data_type = 'DeltaG'
        alias = data_type
        profile = deltaG_profile_per_sequence(seq=self.seq, w=self.w, alias=alias)
        profile = profile / profile.min()
        profile['position'] = profile.index + 1
        dfs.append(profile)

        mat = reduce(lambda left, right: pd.merge(left, right, on=['position']), dfs)
        if self.matrix is not None:
            print("WARNING: overriding existing train etropy matrix")
        self.matrix = mat
        return mat


    def generate_data(self):
        """
        process final input matrix
        :return: a data frame containing cluster assignments
        """
        self.get_profiles()
        self.matrix['seq_id'] = self.alias
        self.matrix['family'] = self.family
        if self.out != None:
            self.matrix.to_csv(os.path.join(self.out, f'input_mat_{self.alias}.csv'), index=False)