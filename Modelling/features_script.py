import pandas as pd
import numpy as np
import os
import sys
print(sys.path)
sys.path.append('/home/jovyan/work/')
from ExploratoryAnalysis.data_processing_modules import *
from ExploratoryAnalysis.clustering_modules import *
from tqdm import tqdm

def load_training():
    full = pd.DataFrame()
    for i in range(2):
        df = pd.read_csv(f"../MIND_large/80_20/train_chunk{i}.csv", index_col=0)
        full = pd.concat([full, df])
    return full

train_tf = load_training()

user = apply_transformations(train_tf)
user.to_csv("../MIND_large/80_20/user_prof_train.csv")

metrics = ['euclidean','cosine']
parameters = [(0.0, 30),(0.1, 30),(0.0,50),(0.1,50)]
embeddings = []
fname = "../MIND_large/80_20/user_embeddings_"

for metric in tqdm(metrics, total=2):
    for p_comb in tqdm(parameters, total=8):
        min_dist, n_neigh = p_comb
        embeddings = create_UMAP_embeddings(2, user.iloc[:,1:], metric, n_neigh, min_dist, 200)
        np.save(fname + f'{metric}_{min_dist}_{n_neigh}', embeddings)

