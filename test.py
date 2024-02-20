from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt


# Dimension reduction and clustering libraries
import umap.umap_ as umap
import sklearn.cluster as cluster
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score 
meep = umap.UMAP().fit_transform(np.random.randn(10, 3))
print(meep)