import pandas as pd 
import numpy as np

import umap.umap_ as umap

import matplotlib.pyplot as plt
import matplotlib
import plotly.express as px
import sklearn.cluster as cluster

from clustering_modules import create_UMAP_embeddings

user = pd.read_csv("../MIND_large/csv/user_features.csv", index_col=0)
metrics = ['euclidean','cosine']
parameters = [(0.0, 30),(0.1, 30),(0.0,50)]
embeddings = []
fname = "../MIND_large/csv/user_embeddings_"
for metric in metrics:
    for p_comb in parameters:
        min_dist, n_neigh = p_comb
        embeddings = create_UMAP_embeddings(2, user.iloc[:,1:], metric, min_dist, n_neigh)
        np.save(fname + f'{metric}_{min_dist}_{n_neigh}', embeddings)
