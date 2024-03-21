import matrix_modules 
import pandas as pd
import numpy as np

import sys
sys.path.append('/home/jovyan/work/')
import ExploratoryAnalysis.clustering as cm

# Path handling
import os
module_dir = os.path.dirname(__file__)
data_path = os.path.join(module_dir, '../MIND_large/csv')
emb_path = os.path.join(module_dir, '../MIND_large/embeddings')
"""
This script runs gradient descent, alternating least squares and factorization machines
with several different parameters to determine their affects on performance.
"""

## Loading data and getting things ready

# Load in the full ratings data with news and users.
print("Loading in data")
full_ratings, news, users = matrix_modules.load_dataset_for_matrix()


# Create a ratings matrix R, and item and user index hash maps for easy subsetting.
als_items = [matrix_modules.create_item_cluster_mat(full_ratings, news, isALS=True, num_users=len(users), num_clusters=len(news['cluster'].unique())),
         matrix_modules.create_user_cluster_mat(full_ratings, news, users, isALS=True, num_user_clusters=len(users['cluster'].unique()))]


# # items is now of the form [(R, user_idx, item_idx)]
# counter = 0
# results = pd.DataFrame()
# print("Starting Training Loop")
# for i in range(2):

#     if i == 0:
#         clustering = "item"
#         cluster_type = False
#     else:
#         clustering = "user"
#         cluster_type = True

#     print("Unpacking info")
#     R, D, item_idx, user_idx = als_items[i]

#     # Make the indices into lists and sort them.
#     item_idx = {num : sorted(list(users)) for num, users in item_idx.items()}
#     user_idx = {user_id : sorted(list(ratings)) for user_id, ratings in user_idx.items()}

#     # Create feature matrices for testing.
#     # user_features = pd.read_csv("../MIND_large/csv/user_features.csv", index_col=0)
#     # item_features = pd.read_csv("../MIND_large/csv/item_features.csv", index_col=0)

#     for k in [5, 25, 50]:
#         print(f"Starting for K equal {k}")
#         for lambda_reg in [1, 5]: 
#             print(f"Starting for K equal {k} and lambda equal {lambda_reg}")
#             for max_iteration in [10]:
                    
#                 # Initialize U and V
#                 K = k # Here is where we choose the number of latent factors we would like to include in our matrices.
#                 I = len(user_idx) # number of users
#                 M = len(item_idx) # number of items
#                 U = np.random.uniform(0, 1, size=K*I).reshape((K ,I))
#                 V = np.random.uniform(0, 1, size=K*M).reshape((K, M))

#                 U, V, track_error, track_update = matrix_modules.alternating_least_squares(U, V, R, user_idx, item_idx, max_iterations=max_iteration, lambda_reg=lambda_reg)
#                 param_info = pd.DataFrame(data={"clustering_type": clustering, "alg" : "ALS", "k" : k, "lambda_reg" : lambda_reg, 'added_features' : i, 'RMSE' : [track_error], 'Max Updates' : [track_update]}, index=[counter])
#                 counter += 1
#                 results = pd.concat([results, param_info], axis=0)
            

# results.to_csv(data_path + "/als_testing_output.csv")

print("Loading data for SGD")


counter = 0
results = pd.DataFrame()
print("Starting Training Loop")
for i in range(2):

    if i == 0:
        clustering = "item"
        cluster_type = False
    else:
        clustering = "user"
        cluster_type = True

    print("Unpacking info")
    R, D, item_idx, user_idx = als_items[i]

    # Make the indices into lists and sort them.
    item_idx = {num : sorted(list(users)) for num, users in item_idx.items()}
    user_idx = {user_id : sorted(list(ratings)) for user_id, ratings in user_idx.items()}

    # Create feature matrices for testing.
    # user_features = pd.read_csv("../MIND_large/csv/user_features.csv", index_col=0)
    # item_features = pd.read_csv("../MIND_large/csv/item_features.csv", index_col=0)
    for j in range(42, 48):
        np.random.seed(j)
        for k in [5, 25, 50]:
            print(f"Starting for K equal {k}")
            for lambda_reg in [1, 5]: 
                print(f"Starting for K equal {k} and lambda equal {lambda_reg}")
                for max_iteration in [100]:

                    # Initialize U and V
                    K = k # Here is where we choose the number of latent factors we would like to include in our matrices.
                    I = len(user_idx) # number of users
                    M = len(item_idx) # number of items
                    U = np.random.uniform(0, 1, size=K*I).reshape((I, K))
                    V = np.random.uniform(0, 1, size=K*M).reshape((M, K))

                    U, V, track_error, track_update = matrix_modules.vectorized_gradient_descent(R, U, V, D, rate=0.0000001, max_iterations=max_iteration, lam=lambda_reg)
                    param_info = pd.DataFrame(data={"seed": j, "clustering_type": clustering, "alg" : "SGD", "k" : k, "lambda_reg" : lambda_reg, 'added_features' : i, 'RMSE' : [track_error], 'Max Updates' : [track_update]}, index=[counter])
                    counter += 1
                    results = pd.concat([results, param_info], axis=0)

results.to_csv(data_path + "/gd_testing_output.csv")