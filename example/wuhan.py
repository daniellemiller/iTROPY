from Bio import SeqIO
import os, pickle

import src.profiler as pf
import src.drops as dp

fasta_wuhan = r'/Users/daniellemiller/Google Drive/Msc Bioinformatics/Projects/Entropy/data/Wuhan/gisaid_cov2020_all_sequences.fasta'
fasta_mers = r'/Users/daniellemiller/Google Drive/Msc Bioinformatics/Projects/Entropy/data/Wuhan/MERS-CoV.fasta'
fasta_sars = r'/Users/daniellemiller/Google Drive/Msc Bioinformatics/Projects/Entropy/data/Wuhan/SARS_CoV_AY278741.fasta'

seq = ''
for rec in SeqIO.parse(fasta_sars, 'fasta'):
    seq = str(rec.seq)

out = r'/Users/daniellemiller/Google Drive/Msc Bioinformatics/Projects/Entropy/data/Wuhan/'
#alias = rec.name.replace('|', '_').replace('/', '_').replace('-', '_')
#alias = rec.name + "_MERS_CoV_KOR_KNIH_002_05_2015"
alias = 'SARS_CoV_AY278741'

profiler = pf.Profiler(seq, out_dir=out, w=200, l=100, alias=alias, family='Coronaviridae')
profiler.generate_data()

drops = dp.Drops(profiler, k=5)
drops.get_drops()

drops.update_profiler()
drops.categorize_drops()
drops.rank_drops()
drops.dropplot()

