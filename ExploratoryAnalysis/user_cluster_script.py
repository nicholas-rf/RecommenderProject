import pandas as pd 
import numpy as np
from ExploratoryAnalysis.clustering import create_UMAP_embeddings
import os

"""
This script is ran to generate and save several different 
"""

# Path handling
module_dir = os.path.dirname(__file__) 
data_path = os.path.join(module_dir, '../MIND_large/csv')
emb_path = os.path.join(module_dir, '../MIND_large/embeddings')

# Read in the user features.
user = pd.read_csv(data_path + "/user_features.csv", index_col=0)

# Create lists of metrics to experiment on.
metrics = ['euclidean','cosine']
# parameters = [(0.0, 30),(0.1, 30),(0.0,50)]

parameters = [(0.1,50)]
# Initialize a standard filename to append to when saving files.
fname = emb_path + "/user_embeddings_"

# Iterate over euclidean and cosine distance metrics and each parameter set in parameters.
for metric in metrics:
    for p_comb in parameters:

        # Unpack the parameters.
        min_dist, n_neigh = p_comb

        # Load the parameters into the create_UMAP_embeddings function, create embeddings and then save them.
        embeddings = create_UMAP_embeddings(2, user.iloc[:,1:], metric, n_neighbors=n_neigh, min_dist=min_dist)
        np.save(fname + f'{metric}_{min_dist}_{n_neigh}', embeddings)
