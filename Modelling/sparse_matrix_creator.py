from sklearn.feature_extraction import DictVectorizer
import matrix_modules
from tqdm import tqdm
import numpy as np
import pandas as pd
from scipy import sparse

# loads full tensorflow dataframe
dataset = matrix_modules.load_in_tensorflow_full()

label_list = ["unc", "unct", "uncts"]
list_of_lists = [["user_id", "news_id","category"],
                 ["user_id", "news_id","category","time"],
                 ["user_id", "news_id","category","time","sub_category"]]

for i,feature_list in tqdm(enumerate(list_of_lists)):
    v = DictVectorizer()
    x = v.fit_transform(list(dataset[feature_list].T.to_dict().values()))

    sparse.save_npz(f"../MIND_large/embeddings/sparse_mat_fm_{label_list[i]}.npz", x)