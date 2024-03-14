import os
import matrix_modules 
import pandas as pd
import numpy as np
np.random.seed(42)
import sys
sys.path.append('/home/jovyan/work/')
import ExploratoryAnalysis.clustering_modules as cm

"""
This script runs stochastic gradient descent, alternating least squares and factorization machines
with several different parameters to determine their affects on performance.
"""

## Loading data and getting things ready

# Load in the full ratings data with news and users.
print("Loading in data")
full_ratings, news, users = matrix_modules.load_dataset(full=True)


# Create a ratings matrix R, and item and user index hash maps for easy subsetting.
items = [matrix_modules.create_item_cluster_mat(full_ratings, news, isALS=True, num_users=len(users), num_clusters=len(news['cluster'].unique())),
         matrix_modules.create_user_cluster_mat(full_ratings, news, users, isALS=True, num_user_clusters=len(users['cluster'].unique()))]

del news, users

# items is now of the form [(R, user_idx, item_idx)]
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
    R, item_idx, user_idx = items[i]

    # Make the indices into lists and sort them.
    item_idx = {num : sorted(list(users)) for num, users in item_idx.items()}
    user_idx = {user_id : sorted(list(ratings)) for user_id, ratings in user_idx.items()}

    # Create feature matrices for testing.
    user_features = pd.read_csv("../MIND_large/csv/full_user_clusters.csv", index_col=0)
    item_features = pd.read_csv("../MIND_large/csv/full_item_features.csv", index_col=0)

    for umapDim in range(1, 11):
    
        user_features_new, item_features_new = matrix_modules.create_features(user_features, item_features, userClustering=cluster_type, umapDim=umapDim)
        
        for k in range(5, 55, 5):

            # Initialize U and V
            K = k # Here is where we choose the number of latent factors we would like to include in our matrices.
            I = len(user_idx) # number of users
            M = len(item_idx) # number of items
            U = np.random.uniform(0, 1, size=K*I).reshape((K, I))
            V = np.random.uniform(0, 1, size=K*M).reshape((K, M))
            U = np.concatenate((U, user_features_new.T), axis=0)
            V = np.concatenate((V, item_features_new.T), axis=0)
            
            for lambda_reg in [0.01, 0.05, 0.10, 0.15, 0.20]:
                
                for max_iteration in [10, 100, 1000, 10000]:
                    
                    U, V, track_error, track_update = matrix_modules.alternating_least_squares(U, V, R, user_idx, item_idx, max_iterations=max_iteration, lambda_reg=lambda_reg)
                    errors = [rmse for rmse in [track_error[i]['rmse'] for i in range(len(track_error))]]
                    max_updates = [update for update in [track_update[i]['max_update'] for i in range(len(track_update))]]
                    print(errors)
                    print(max_updates)
                    param_info = pd.DataFrame(data={"clustering_type": clustering, "alg" : "ALS", "k" : k, "lambda_reg" : lambda_reg, "umapDim": umapDim, 'RMSE' : [errors], 'Max Updates' : [max_updates]}, index=[counter])
                    counter += 1
                    results = pd.concat([results, param_info], axis=0)
                    print(results)

results.to_csv("../MIND_large/csv/testing_output.csv")
# SAMPLE CODE FROM FINAL PROJECT WHICH WE WILL BE USING FOR TRACKING MODEL PERFORMANCE
# track_rmse += [{
#     'iteration':i, 
#     'rmse': rmse(Gnew),
#     'max residual change': max_update(Gnew, G, relative=False)
# }]
# track_update += [{
#     'iteration':i, 
#     'max update':max(max_update(Unew, U), max_update(Vnew, V))
# }]


