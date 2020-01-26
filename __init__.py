# import itropy objects

__author__ = """Danielle Miller"""
__version__ = '0.0.1'

import src.profiler as pf
import src.drops as dp

__docformat__ = "restructuredtext"


# Let users know if they're missing any module
hard_dependencies = ("numpy", "RNA", "tqdm", "Bio", "sklearn", "statsmodels", "seaborn", "matplotlib")
missing_dependencies = []

for dependency in hard_dependencies:
    try:
        __import__(dependency)
    except ImportError as e:
        missing_dependencies.append("{0}: {1}".format(dependency, str(e)))

if missing_dependencies:
    raise ImportError(
        "Unable to import required dependencies:\n" + "\n".join(missing_dependencies))


__doc__ = "type here doc"


def fit_profiles(seq, out_dir=None, w=200, l =100, alias='Seq', family='Unknown', k=5):
    """
    Fit entropy and deltaG profiles to a given sequence.
    :param seq: a sequence of type string.
    :param out_dir: a defined output dir to save csv files to.
    :param w: window size for profile generation, default=20.
    :param l: window size for permutation test on drop sizes. default=100
    :param alias: sequence ID. type string. optional. default='Seq'
    :param family: the family taxonomy of a genetic sequence. default ='Unknown'.
    :param k: the kmer size for entropy calculation per window. default=5 (k>7 is computationally heavy)
    :return: fits the entropy profile and generate drops entropy drops
    """
    profiler = pf.Profiler(seq, out_dir, w, l, alias, family)
    profiler.generate_data()
    drops = dp.Drops(profiler, k)
    drops.get_drops()

    return drops

def predict(drops):
    """
    predict drop rank and category: Repetitive\ Structural
    :param drops: a Drop object after data generation and
    :return: Drop object, categorized and ranked.
    """
    if drops.drops is None:
        raise Exception("Drops data is not available")
    if drops.X is None:
        raise Exception("Train data for prediction is not available")

    # Predict drop type and drop rank
    drops.update_profiler()
    drops.categorize_drops()
    drops.rank_drops()
    return drops


