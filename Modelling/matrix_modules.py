import pandas as pd
import numpy as np
from tqdm import tqdm
import sys
sys.path.append('/home/jovyan/work/')
import ExploratoryAnalysis.clustering as cm

def load_in_tensorflow_full():
    """
    Loads in the full tensorflow compatible dataset for usage within matrix factorization or tensorflow modelling.
    """
    # Load in the complete Tensorflow compatible dataset by iterating over all the chunks.
    dataset = pd.DataFrame()
    for i in range(4):
        df = pd.read_csv(f"../MIND_large/csv/tensorflow_dataset_chunk{i}.csv", index_col=0)
        dataset = pd.concat([dataset, df])
    return dataset

def load_dataset_for_matrix():
    """
    Loads in the full Tensorflow compatible dataset with clustered news and users datasets. 
    More specifically the Tensorflow compatible dataset grouped by user ids containing lists of scores and seen news. 
    
    Args:
        None 

    Returns:
        (pd.DataFrame, pd.DataFrame, pd.DataFrame) : Compatible dataset for matrix creation, the news dataset with cluster labels, 
            and the complete users clustered dataset. 
    """
    
    # Load in the full Tensorflow compatible dataset.
    dataset = load_in_tensorflow_full()

    # Load in the clustered news and users datasets
    news = pd.read_csv('../MIND_large/csv/item_features.csv')
    all_ratings = dataset.groupby('user_id')['news_id'].apply(list).reset_index()
    scores = dataset.groupby('user_id')['score'].apply(list).reset_index()
    all_ratings['scores'] = scores['score']

    user_clustered = pd.read_csv('../MIND_large/csv/user_features.csv') # needs to be updated for the train test split as well :D 
    
    return all_ratings, news, user_clustered

def create_item_cluster_mat(ratings_df, news, num_users, num_clusters, isALS=False):
    """
    Creates a user item ratings matrix R with item clusters and potentially associated hash maps for use within ALS matrix factorization. Due to the nature of ALS, the 
    item indices that a user has interacted and the user indices that have interacted with an item get used heavily so these hash maps
    are initialized during construction of R. This is a modified version of a similar function to deal with item clusters.

    Args:
        ratings_df (pd.DataFrame) : A pandas dataframe containing the result of grouping the dataset by userID, and applying
            lists to both the articles they interacted with and their scores.
        news (pd.DataFrame) : The news dataset with labels.
        num_users (int) : The number of users that are being used for the matrix, gets used to determine the number of rows
            necessary for the hash table that is used to make the matrix. Also gets used to generate ALS specific hash tables
            for efficient subsetting of the ratings, item and user feature matrices.
        num_clusters (int) : The number of clusters created for the items to be used in generating the number of columns 
            necessary for the hash table that is used to make the matrix. Also gets used to generate ALS specific hash tables
            for efficient subsetting of the ratings, item and user feature matrices.
        isALS (bool) : A boolean to determine if hash tables to store extra values should be generated for the ALS matrix factorization algorithm.
    
    Returns:
        If isALS is True,
        np.column_stack(list(matrix.values())), cluster_idx, user_idx (np.2darray, dict, dict) : The ratings matrix and resulting ALS specific hash tables for 
            column : row values and row : column values if isALS is true. 
        If isALS is False,
        np.column_stack(list(matrix.values())) (np.2darray) : The ratings matrix.
        np.column_stack(list(d_mat.values())) (no.2darray) : The seen matrix which represents D for more efficient GD.
        
    """
    # Initialize the hash map that will create the matrix as a list of np zero arrays for each cluster and the hashmap of item clusters .
    item_clusters = {item : cluster for item, cluster in zip(news['news_id'], news['cluster'])}

    matrix = {cluster : np.full(num_users, 0, dtype='int16') for cluster in range(num_clusters)} 
    d_mat =  {cluster : np.full(num_users, 0, dtype='int16') for cluster in range(num_clusters)}

    if isALS:
        # Initialize the cluster hashmap, which is used to track an item index and all row indices that engage with that item.
        cluster_idx = {cluster : set() for cluster in range(num_clusters)}

        # Initialize the user hashmap, which is used to track a user index and all item indices that the user engaged with.
        user_idx = {user_id : set() for user_id in range(num_users)}

    # Initialize a counter to keep track of user index.
    counter = 0

    # Iterate over every user, their ratings and score in the rating matrix.
    for user, ratings, score in zip(ratings_df['user_id'], ratings_df['news_id'], ratings_df['scores']):
        for index in range(len(ratings)):
            # Get the news id of the interaction and their rating.
            news_id = ratings[index]
            num = score[index]

            # If the rating is not zero, add a 1 to the cluster of the article.
            if num != 0:    
                matrix[item_clusters[news_id]][counter] += 1
                d_mat[item_clusters[news_id]][counter] = 1
                
                # If we are using ALS, add relevant indices to the hashmaps.
                if isALS:
                    cluster_idx[item_clusters[news_id]].add(counter)
                    user_idx[counter].add(item_clusters[news_id])
        counter += 1

    # If we are not making hash maps for ALS, then only return the matrix. Otherwise return the matrix and hash maps.

    return np.column_stack(list(matrix.values())), np.column_stack(list(d_mat.values())), cluster_idx, user_idx

def create_user_cluster_mat(ratings_df, news, user_clustered, num_user_clusters, isALS=False):
    """
    Creates a user item ratings matrix R with user clusters and potentially associated hash maps for use within ALS matrix factorization. Due to the nature of ALS, the 
    item indices that a user has interacted and the user indices that have interacted with an item get used heavily so these hash maps
    are initialized during construction of R. This is a modified version of a similar function to deal with user clusters.

    Args:
        ratings_df (pd.DataFrame) : A pandas dataframe containing the result of grouping the dataset by userID, and applying
            lists to both the articles they interacted with and their scores.
        news (pd.DataFrame) : The news dataset with clustering labels.
        num_users (int) : The number of users that are being used for the matrix, gets used to determine the number of rows
            necessary for the hash table that is used to make the matrix. Also gets used to generate ALS specific hash tables
            for efficient subsetting of the ratings, item and user feature matrices.
        num_clusters (int) : The number of clusters created for the items to be used in generating the number of columns 
            necessary for the hash table that is used to make the matrix. Also gets used to generate ALS specific hash tables
            for efficient subsetting of the ratings, item and user feature matrices.
        isALS (bool) : A boolean to determine if hash tables to store extra values should be generated for the ALS matrix factorization algorithm.
    
    Returns:
        If isALS is True,
        np.column_stack(list(matrix.values())), cluster_idx, item_idx (np.2darray, dict, dict) : The ratings matrix and resulting ALS specific hash tables for 
            column : row values and row : column values if isALS is true. 
        If isALS is False,
        np.column_stack(list(matrix.values())) (np.2darray) : The ratings matrix.
        np.column_stack(list(d_mat.values())) (no.2darray) : The seen matrix which represents D for more efficient GD.
    """
    # Initialize the hash map that will create the matrix as a list of np zero arrays for each news_id and the hashmap of user clusters.
    matrix = {news_id : np.full(num_user_clusters, 0, dtype='int8') for news_id in news['news_id']}
    d_mat =  {news_id : np.full(num_user_clusters, 0, dtype='int8') for news_id in news['news_id']}
    user_clusters = {user : cluster for user, cluster in zip(user_clustered['user_id'], user_clustered['cluster'])}
    
    if isALS:
        # Initialize the item idx hash map to create a hash map that can be used to check all row indices that have an appearance in the column.
        item_lookup = {news_id : index for index, news_id in enumerate(news['news_id'])}
        item_idx = {index : set() for index in range(len(news['news_id']))}

        # Initialize the cluster idx hash map to create a hash map that can be used to check all column indicies that have an appearance in the row.
        cluster_idx = {cluster : set() for cluster in range(num_user_clusters)}
    
    # Iterate over the user ids, ratings and scores.
    for user, ratings, score in zip(ratings_df['user_id'], ratings_df['news_id'], ratings_df['scores']):
        
        # Determine the users cluster and then iterate all of their ratings.
        cluster = user_clusters[user]
        for index in range(len(ratings)):
            # Get the news ID and score of their rating.
            news_id = ratings[index]
            num = score[index]

            # If the score is not zero, add 1 to the clusters score for that item.
            if num != 0:
                # Access the column of the matrix, and then find the index in that column for that users cluster, then increment by 1.
                matrix[news_id][cluster] += 1 
                d_mat[news_id][cluster] = 1

                # If we are using ALS, add the relevant indices to the hash map.
                if isALS:
                    # Add the corresponding cluster number to the column index for the news id.
                    item_idx[item_lookup[news_id]].add(cluster)

                    # Add the column index to the clusters key.
                    cluster_idx[cluster].add(item_lookup[news_id])

    # If we are not making hash maps for ALS, then only return the matrix. Otherwise return the matrix and hash maps.                      
    
    return np.column_stack(list(matrix.values())), np.column_stack(list(d_mat.values())), item_idx, cluster_idx
    
def get_features(user_features, item_features, userClustering=False, umapDim=2):
    """
    Uses UMAP to reduce the dimension of user and item features to a dimension specified by umapDim. By reducing the dimension
    we standardize the number of features being added to U and V.
        
    Args: 
        user_features (pd.DataFrame) : The user features in a pandas dataframe.
        item_features (pd.DataFrame) : The item features in a pandas dataframe.
        isALS (bool) : Signifies if the process is being done for alternating least squares or not as the matrix dimensions are different for SGD and ALS.
    
    Returns:
        user_features (pd.DataFrame) : The user features in a 2d numpy array as dimension reduced embeddings.
        item_features (pd.DataFrame) : The item features in a 2d numpy array as dimension reduced embeddings.
    """
    
    # If we are using clustering we need to aggregate features per cluster so they can be properly added to U and V.
    if userClustering:
        # Group by the cluster label, apply the sum function to get a groups features, and then drop the user_id.
        user_features = user_features.groupby('cluster').agg(sum).drop(columns=['user_id'])  

        # Drop the news_id from the item features.
        item_features.drop(columns="news_id", inplace=True)

        # Reduce the features to a consistent size such that the matrices have the same number of latent factors.
        user_features = cm.create_UMAP_embeddings(umapDim, user_features, metric='euclidean', n_neighbors=10)
        item_features = cm.create_UMAP_embeddings(umapDim, item_features, metric='euclidean', n_neighbors=50)

    else:
        # Group by the cluster label, apply the sum function to get a groups features, and then drop the news_id, and reduced embeddings.
        item_features = item_features.groupby('cluster').agg(sum).drop(columns=['news_id', "reduced_embeddings_1", "reduced_embeddings_2"])
        
        # Drop the user id from the user features.
        user_features.drop(columns="user_id", inplace=True)

        # Reduce the features to a consistent size such that the matrices have the same number of latent factors.
        item_features = cm.create_UMAP_embeddings(umapDim, item_features, metric='euclidean', n_neighbors=10)
        user_features = cm.create_UMAP_embeddings(umapDim, user_features, metric='euclidean', n_neighbors=70)

    return user_features, item_features

def rmse(X):
    """
    Computes root-mean-square-error, ignoring nan values
    """
    mask = X != 0
    return np.sqrt(np.nanmean((X[mask])**2))

def max_update(X, Y, relative=True):
    """
    Compute elementwise maximum update
    
    parameters:
    - X, Y: numpy arrays or vectors
    - relative: [True] compute relative magnitudes
    
    returns
    - maximum difference between X and Y (relative to Y) 
    
    """
    if relative:
        updates = np.nan_to_num((X - Y)/Y)
    else:
        updates = np.nan_to_num(X - Y)
            
    return np.linalg.norm(updates.ravel(), np.inf)

def vectorized_gradient_descent(R, U, V, D, rate=0.00001,max_iterations=10,lam=5, diff_threshold=1e-3):
    """
    Performes vectorized gradient descent to make ratings predictions for the incomplete user-item matrix. Compared to standard gradient descent, 
    vectorized gradient descent further improves upon update formulae by using matrices Gamma and D to operate on U and V for observed ratings 
    instead of iterating over all observed indices.

    Args:
        R (np.ndarray) : The user-item ratings matrix R.
        U (np.ndarray) : The user latent factor matrix U.
        V (np.ndarray) : The item latent factor matrix V.
        D (np.ndarray) : The observed interaction matrix D.
        rate (float) : The learning rate for the gradient descent. Here we need to use small values like 0.00001 due to 
            the numerical instability of our data brought on by clustering.
        max_iterations (int) : The number of iterations to run gradient descent for.
        lam (float) : The regularization parameter which adds a penalty to vectors with large magnitude.
        diff_threshold (float) : The threshold to stop iterating over the data with to avoid computational innefficiency.
    """

    # Create initial Uold and Vold for calculating max updates.
    Uold = np.zeros_like(U)
    Vold = np.zeros_like(V)
    
    # Initialize empty lists to track error and update.
    error_update = []
    track_update = []

    # Perform gradient descent for the number of iterations as specified by max_iterations.
    for t in tqdm(range(1, max_iterations+1), total=max_iterations, desc='Running Optimized Gradient Descent'): # , total=max_iterations, desc="Starting descent"):

        # Create Gamma as the residuals matrix.
        Gamma = R - (U @ V.T)

        # Take the hadamard product of gamma and matrix D to get a matrix of all observed residuals.
        observed_errors = np.multiply(Gamma, D)

        # Update the entire matrix U with vectorized operations.
        U += rate * (observed_errors @ V - lam * U)
        
        # Find Gamma and observed_errors with the updated U.    
        Gamma = R-(U @ V.T)
        observed_errors = np.multiply(Gamma, D)

        # Update the entire matrix V with vectorized operations.
        V += rate * (observed_errors.T @ U - lam * V)

        observed_errors = np.multiply(Gamma, D)

        # Update the update tracking.    
        track_update += [
            max(max_update(U, Uold), max_update(V, Vold))
        ]

        # If our most recent update is lower than our difference threshold, return the Old matrices and information arrays.
        if track_update[-1] < diff_threshold:
            print("Threshold reached, stopping descent")
            return Uold, Vold , error_update, track_update

        # Update Uold and Vold.
        Uold = U.copy() 
        Vold = V.copy()

        # Update the error.
        error_update += [rmse(observed_errors)]
    
    # Return the new U and V and update lists. 
    return U, V , error_update, track_update

def alternating_least_squares(U, V, R, user_map, item_map, max_iterations=10, lambda_reg=0.01, diff_threshold = 1e-3):
    """
    Takes in the ratings matrix, user and item matrices, and performes alternating least squares optimization for iterations
    determined by max_iterations regularized by lambda_reg.

    Args:
        U (np.ndarray) : The k x n user feature matrix.
        V (np.ndarray) : The k x m item feature matrix.
        R (np.ndarray) : The ratings matrix.
        user_map (dict) : The hash map containing user ids as keys and item indices as values, gets used to subset the ratings matrix and V.
        item_map (dict) : The hash map containing item ids as keys and user indices as values, gets used to subset the ratings matrix and U.
        max_iterations (int) : The number of iterations to run alternating least squares for.
        lambda_reg (float) : The regularization term in the alternating least squares algorithm.

    Returns:
        U (np.ndarray) : The optimized user feature matrix.
        V (np.ndarray) : The optimized item feature matrix.
    """
    
    # Initialize old versions of U and V to track max updates.
    Uold = np.zeros_like(U)
    Vold = np.zeros_like(V)

    # Initialize RMSE and Max update lists for tracking
    track_error = [rmse(R - U.T @ V)]
    track_update = [max(max_update(U, Uold), max_update(V, Vold))]

    # Initialize k and the number of columns in each matrix
    k, u_cols = U.shape
    _, v_cols = V.shape
    k_In = np.diag(np.full(k, lambda_reg))

    # Start optimizing U and V
    for iteration in tqdm(range(1, max_iterations+1), total=max_iterations, desc='Starting ALS iterations'):
        # Fix V and optimize U 
        Uold = U.copy()
        Vold = V.copy()
        for i in tqdm(range(u_cols), total=u_cols, desc='Optimizing U', leave=False):
            # Using translator inverse here to make sure we are using small matrix
            ratings_row = R[i, user_map[i]]
            rated_items = V[:, user_map[i]]

            # Update the ith vector of U
            U[:, i] = np.linalg.inv((rated_items @ rated_items.T) + k_In) @ (ratings_row @ rated_items.T)

        # Fix U and optimize V
        for j in tqdm(range(v_cols), total=v_cols, desc='Optimizing V', leave=False):
            # Get the ratings for the item 
            ratings_row = R[item_map[j], j]
            user_features = U[:, item_map[j]]

            # Update the jth vector of V
            V[:, j] = np.linalg.inv((user_features @ user_features.T) + k_In) @ (ratings_row @ user_features.T)
        
        
        
        # Calculate the error and maximum update for this iteration
        track_error += [rmse(R - (U.T @ V))]
        track_update += [max(max_update(U, Uold), max_update(V, Vold))]
        Uold = U.copy()
        Vold = V.copy()
        # If our most recent update is lower than our difference threshold, return the Old matrices and information arrays.
        if track_update[-1] < diff_threshold:
            print("Threshold reached, stopping alternating squares")
            return Uold, Vold , track_error, track_update

    return U, V, track_error, track_update