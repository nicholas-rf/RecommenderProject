import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys
import seaborn as sns
sys.path.append('/home/jovyan/work/')
import ExploratoryAnalysis.clustering as cm
from sklearn.neighbors import NearestNeighbors

# Path handling
import os
module_dir = os.path.dirname(__file__)
data_path = os.path.join(module_dir, '../MIND_large/csv')
emb_path = os.path.join(module_dir, '../MIND_large/embeddings')

def load_in_tensorflow_full():
    """
    Loads in the full tensorflow compatible dataset for usage within matrix factorization or tensorflow modelling.
    """
    # Load in the complete Tensorflow compatible dataset by iterating over all the chunks.
    dataset = pd.DataFrame()
    for i in range(4):
        df = pd.read_csv(data_path + f"/tensorflow_dataset_chunk{i}.csv", index_col=0)
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
    news = pd.read_csv(data_path + '/item_features.csv')
    all_ratings = dataset.groupby('user_id')['news_id'].apply(list).reset_index()
    scores = dataset.groupby('user_id')['score'].apply(list).reset_index()
    all_ratings['scores'] = scores['score']

    user_clustered = pd.read_csv(data_path + '/user_features.csv') # needs to be updated for the train test split as well :D 
    
    return all_ratings, news, user_clustered

def create_item_cluster_mat(ratings_df, news, num_users, num_clusters):
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

    Returns:
        np.column_stack(list(matrix.values())),
        np.column_stack(list(d_mat.values())),
        cluster_idx, item_idx (np.2darray, dict, dict) : The ratings matrix, the observed matrix and hash tables.

    """
    # Initialize the hash map that will create the matrix as a list of np zero arrays for each cluster and the hashmap of item clusters .
    item_clusters = {item : cluster for item, cluster in zip(news['news_id'], news['cluster'])}

    matrix = {cluster : np.full(num_users, 0, dtype='int16') for cluster in range(num_clusters)} 
    d_mat =  {cluster : np.full(num_users, 0, dtype='int16') for cluster in range(num_clusters)}


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

                cluster_idx[item_clusters[news_id]].add(counter)
                user_idx[counter].add(item_clusters[news_id])
        counter += 1

    # If we are not making hash maps for ALS, then only return the matrix. Otherwise return the matrix and hash maps.

    return np.column_stack(list(matrix.values())), np.column_stack(list(d_mat.values())), cluster_idx, user_idx

def create_user_cluster_mat(ratings_df, news, user_clustered, num_user_clusters):
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
        np.column_stack(list(matrix.values())),
        np.column_stack(list(d_mat.values())),
        cluster_idx, item_idx (np.2darray, dict, dict) : The ratings matrix, the observed matrix and hash tables.

    """
    # Initialize the hash map that will create the matrix as a list of np zero arrays for each news_id and the hashmap of user clusters.
    matrix = {news_id : np.full(num_user_clusters, 0, dtype='int8') for news_id in news['news_id']}
    d_mat =  {news_id : np.full(num_user_clusters, 0, dtype='int8') for news_id in news['news_id']}
    user_clusters = {user : cluster for user, cluster in zip(user_clustered['user_id'], user_clustered['cluster'])}
    

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

def vectorized_gradient_descent(R, U, V, D, rate=0.00001, max_iterations=10,lam=5, diff_threshold=1e-3):
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
        if iteration > 3:
            if abs(track_error[-1] - track_error[-2]) < diff_threshold:
                print("Threshold reached, stopping alternating squares")
                return U, V , track_error, track_update
        if track_update[-1] < diff_threshold:
            print("Threshold reached, stopping alternating squares")
            return Uold, Vold , track_error, track_update

    return U, V, track_error, track_update

def get_predicted(cluster, alg, k, lam):
    """
    Returns a predicted ratings matrix for use within the recommendation step.

    Args:  
        cluster (str) : Determines the clustering method used.
        alg (str) : Determines whether or not als or gd is used.
        k (int) : The number of latent factors used.
        lam (int) : The regularization parameter for matrix factorization.

    Returns:
        U, V, user_idx, item_idx : The predicted ratings matrix and user and item indices. 
    """
    print("Loading in Data")
    ratings, news, users = load_dataset_for_matrix()

    print("Creating Matrix")
    if cluster == "item":
        R, D, item_idx, user_idx = create_item_cluster_mat(ratings, news, num_users=len(users), num_clusters=len(news['cluster'].unique()))

    if cluster == "user": 
        R, D, item_idx, user_idx = create_user_cluster_mat(ratings, news, users, num_user_clusters=len(users['cluster'].unique()))
    
    item_idx = {num : sorted(list(users)) for num, users in item_idx.items()}
    user_idx = {user_id : sorted(list(ratings)) for user_id, ratings in user_idx.items()}



    print(f"Starting {alg} iterations")
    if alg == "als":
        # Initialize U and V
        K = k # Here is where we choose the number of latent factors we would like to include in our matrices.
        I = len(user_idx) # number of users
        M = len(item_idx) # number of items
        U = np.random.uniform(0, 1, size=K*I).reshape((K, I))
        V = np.random.uniform(0, 1, size=K*M).reshape((K, M))
        U, V, track_error, track_update = alternating_least_squares(U, V, R, user_idx, item_idx, max_iterations=10, lambda_reg=lam)

    if alg == "gd":
        # Initialize U and V
        K = k # Here is where we choose the number of latent factors we would like to include in our matrices.
        I = len(user_idx) # number of users
        M = len(item_idx) # number of items
        U = np.random.uniform(0, 1, size=K*I).reshape((I, K))
        V = np.random.uniform(0, 1, size=K*M).reshape((M, K))
        U, V, track_error, track_update = vectorized_gradient_descent(R, U, V, D, rate=0.000001, max_iterations=10, lam=lam)
    
    return U, V, item_idx, user_idx, ratings

# Untested factorization machine implementation

def y_hat(w_0, x, w, features_matrix):
    """ 
    Calculates the predicted error for the given parameters: w_0, x, w and the subset of V corresponding to the features present in the feature vector.

    Args:
        w_0 (int) : w_0 represents the global bias term added at the beggining of the y_hat calculation.
        x (dict) : The feature vector corresponding to a particular row index. x has two keys, indices and scores, which signify the indices in V and w that have scores and the corresponding scores.
        w (np.ndarray) : w represents the vector of feature weights and is a 1 x num_features dimensional row vector.
        features_matrix (np.ndarray) : features_matrix represents the subsetted matrix of V corresponding to the indices containing values in x.
    
    Returns:
        y_hat (int) : A predicted score for the feature vector.
    """

    # Get the feature indices and their corresponding scores from the feature vector x.
    feature_indices = x["indices"]
    scores = x["scores"]

    # Get the \sum{i=1}^{n}{w_ix_i} term.
    scaled_feature_weights = w[feature_indices] * scores
    
    # Initialize a total for the summation of inner product of pairwise rows in V with scores.
    total = 0

    # Get the number of rows in the subset of the original features matrix for looping.
    rows, _ = features_matrix.shape

    # Loop through all pairwise groups of rows finding their innner product and multiplying by their scores, summing the whole thing.
    for row_1 in range(rows):
        for row_2 in range(rows):
            # Check to see if one row is the same as another as we wish to avoid that case. 
            if row_1 == row_2:
                pass
            else:
                # Add the full calculation to total.
                # Subsetting scores in this way works because the number of scored items directly corresponds to the number of rows in the subset of our feature matrix V.
                total += scores[row_1] * x[row_2] * np.inner(features_matrix[row_1, :], features_matrix[row_2, :])
                
    # Return the linear combination of what we calculate to get the score prediction.
    return w_0 + total + scaled_feature_weights

def update_w_0(w_0, err, alpha):
    """
    Update the global bias term using the error and the learning rate alpha.
    """
    return w_0 + 2 * alpha * err

def FM(x=None,features= ["user_id", "news_id"], n_iter = 50, step_size=1, rank=4):
    """ 
    Takes in or creates sparse matrix of the tensorflow dataset and runs factorization machine on it tracking max update, rmse change, and
    hyperparameter change over time. 
    """
    
    from fastFM import mcmc
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import train_test_split
    # loads full tensorflow dataframe
    dataset = load_in_tensorflow_full()

    if x is None:
      # If no sparse matrix is fed into the function it then creates one using the tensorflow dataset and specified features to be used
      v = DictVectorizer()
      x = v.fit_transform(list(dataset[features].T.to_dict().values()))

    # Creates target y from the score to fit model and find accuracy
    y = np.asarray(dataset["score"])
    # Deletes the tensorflow dataset for memory efficiency
    del dataset

    # Seed for reproduceability
    seed = 123

    # Initial call of FM model does not run or check any parameters just creates the object
    fm = mcmc.FMRegression(n_iter=0, rank=rank, random_state=seed)
    
    # Creates training/testing split
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=42)

    # Allocates and initalizes the model and hyper parameter.
    fm.fit_predict(X_train, y_train, X_test)

    # Initializes list to capture model progress through iterations
    rmse_test = []
    updates = []
    hyper_param = np.zeros((n_iter -1, 3 + 2 * rank), dtype=np.float64)

    for nr, i in tqdm(enumerate(range(1, n_iter)), total = n_iter):
      # Creates V_old variable to measure against for update
      if nr==0:
        V_old = fm.V_
      else:
        V_old = V.copy()
      
      # Predictably change random state to preserve reproduceablility, but prevent local optima and test model stability 
      fm.random_state = i * seed

      # Fit and predict FM model
      y_pred = fm.fit_predict(X_train, y_train, X_test, n_more_iter=step_size)
      # Update progress lists
      V = fm.V_
      updates.append(max_update(V, V_old))
      rmse_test.append(np.sqrt(mean_squared_error(y_pred, y_test)))
      hyper_param[nr, :] = fm.hyper_param_


    return fm.w0_, fm.w_, fm.V_, rmse_test, hyper_param, updates

def get_items(recommendations, item_features, items_df, news, clustering : bool = False):
    """
    Gets the feature representations of the items and a sample of item names if dealing with clusters.
    """
    if clustering == True:
        some_examples = {}
        for key in recommendations:
            some_examples[f'Items from cluster number {key}'] = {"IDs" : [], "Titles" : []}
            cluster_top_4 = items_df[items_df['cluster'] == key].head(4)
            for id, title in zip(cluster_top_4['news_id'], cluster_top_4['title']):
                some_examples[f'Items from cluster number {key}']["IDs"].append(id)
                some_examples[f'Items from cluster number {key}']["Titles"].append(title)
        
        return some_examples
    
    else:
        some_examples = {}
        subset = news[news['news_id']] in item_features.iloc[recommendations, 1] 
        for id, title in zip(subset['news_id'], subset['title']):
            some_examples[f'Items from cluster number {key}']["IDs"].append(id)
            some_examples[f'Items from cluster number {key}']["Titles"].append(title)
        return some_examples

def collaborative_filter(R, user_index, ratings_df, user_features, user_similarity_knn, seen):
    """
    Uses vector similarity to find similar users and their corresponding indices in the ratings matrix
    
    Args:
        R (np.ndarray) : The predicted user item matrix containing ratings.
        user_index (int) : A row index in the ratings matrix corresponding to a user. This is also used to get their features 
            from the user features dataset.
        ratings_df (pd.DataFrame) : The dataframe of ratings grouped by user id loaded in with load_dataset_for_matrix() used to get the users ID.
        user_similarity_knn (sklearn.neighbors.Knearestneighbors (something)) : The knn model fit on user feature vectors to pull similar indices from.
        seen (dict) : A dictionary containing a list of viewed item indices in R for users.
    Returns:
        items_to_rec (set) : A set of item indices to recommend based off of similar users' interactions.
    """
    # Get the users row from the item features data frame.
    user_row = user_features[user_features['user_id'] == ratings_df.iloc[user_index, 0]]

    # Get the indices of similar users in the ratings matrix. 
    _, indices = user_similarity_knn.kneighbors(user_row.iloc[:, 1:])

    # Initialize an empty set to hold onto item indices.
    items_to_rec = set()

    # Iterate over all other user indices.
    for idx in indices[0][1:]:
        
        # Get the user at idx sorted items by using the seen dictionary.
        sorted_rating_indices = np.argsort(R[idx, list(seen[idx])], axis=0)[::-1]

        # For their top 5 items, add it to the set of items to recommend.
        for item in sorted_rating_indices[:5]:
            items_to_rec.add(item)
    
    # Return items to recommend.
    return items_to_rec

def content_filter(user_idx, seen, item_features, item_features_knn):
    """
    Introduces content filtering into the recommendation step by finding similar items to the ones already rated by the user.

    Args:
        user_idx (int) : The users index to get their seen items.
        seen (dict) : The viewed item dictionary.
        item_features (pd.DataFrame) : The item feature dataset used for vector similarity.

    Returns:
        items_to_rec (set) : A set of item indices to recommend based off of similar users' interactions.
    """
    # Initialize an empty set to hold onto item indices.
    items_to_rec = set()

    # Get a list of rated items for the user being recommended to.
    rated = list(seen[user_idx])

    # Iterate over the items finding similar items using the knn model.
    for item_idx in rated:
        _, indices = item_features_knn.kneighbors(item_features.iloc[item_idx, 1:].to_frame().T)

        # Add all similar items to the list of items to recommend.
        for index in indices[0][1:]:
            items_to_rec.add(index)
    
    # Return the list.
    return items_to_rec

def calculate_weights(user_feature_vector, item_indices, item_features):
    """
    Calculates each items rating weight by taking it's features dot product with the users features, then returns the weights.
    
    Args:
        user_feature_vector (pd.DataFrame) : A row from the user features dataframe containing a users features.
        item_indices (set) : A set containing items for recommendation learned through hybrid filtering.
        item_features (pd.DataFrame) : The item features dataframe containing rows for each item and its features.

    Returns:
        weights (np.ndarray) : Returns the weights generated by the feature vector calculations.
    """
    # Take the user feature vector and remove the user_id, cluster and median_time columns, then make it into a numpy array and reshape it.
    user_weights = user_feature_vector.iloc[0, 3:].to_numpy().reshape(-1, 1).T
    row, col = user_weights.shape
    vec = np.zeros((row, col)) + 0.0001

    # Making edge case for users with zero preferences.
    user_weights += vec
    
    # Initialize an empty array for the weights.
    weights = []

    # Iterate over the items that have been collected for the user.
    for item_index in item_indices:

        # Find the items features and take its dot product with the users weights.
        item_weights = item_features.iloc[item_index, 3:].to_numpy().reshape(-1, 1)
        weight = user_weights @ item_weights
        weights.append(weight)
    
    # Return the item weights.
    return weights 

def filer_recommendations(sorted_indices, seen, user_index):
    """
    Filters the recommendations to exclude those that have already been rated by the user
    """
    viewed_items = seen[user_index]
    # Create an empty list to populate with recommendable indices.
    recommendable_items = []

    # Sort through all indices that are recommendable.
    for index in sorted_indices:

        # If the index corresponds with one that has not already been viewed, add it to the list of recommendable items.
        if index not in viewed_items:
            recommendable_items.append(index)
    
    # If the recommendable items list is empty, return the sorted indices. 
    if not recommendable_items and viewed_items:
        recommendable_items = sorted_indices
        print("User has interacted with all possible items")
    if not recommendable_items and not viewed_items:
        if not sorted_indices:
            print("No similar items found!")
            return [1, 2, 3, 4]
        print("User has interacted with no items")
    
    # If the recommendable items list is not empty, return it.
    return recommendable_items

def recommend_items(R, seen, user_index, ratings_df, user_features, item_features, user_similarity_knn, item_features_knn, n, weight_category=None, weight_scale=None):
    """ 
    Recommends items to a user at user_index utilizing hybrid filtering. Collaborative and content filtering utilize separate knn models fitted 
    on feature data for obtaining similar items to those already rated or those that similar users like. These items features are then compared against the users
    features to create weights. Weights are then applied to predicted ratings and the top items are returned.   

    Args:
        R (np.ndarray) : The completed user item matrix.
        seen (dict) : A dictionary containing users and seen items.
        user_index (int) : The index of the user being recommended items.
        ratings_df (pd.DataFrame) : The dataframe of ratings grouped by user id loaded in with load_dataset_for_matrix() used to get the users ID.
        user_features (pd.DataFrame) : The user features dataframe.
        item_features (pd.DataFrame) : The item features dataframe.
        user_similarity_knn (sklearn.neighbors.KNeighborsRegressor) : A fitted knn model on user features for collaborative filtering.
        item_features_knn (sklearn.neighbors.KNeighborsRegressor) : A fitted knn model on item features for content filtering.
        n (int) : The number of items to recommend.
        weight_category (list(str)) : A column or group of columns to add weighting to.
        weight_scale (list(int)) : The weights to apply to each column(s).

    Returns:
        ratings_overall (list) : A list of items to return as recommendations.
    """
    mutable_items = item_features.copy()
    mutable_users = user_features.copy()

    # Use collaborative filtering to obtain items that similar users like.
    collaborative_items = collaborative_filter(R, user_index, ratings_df, mutable_users, user_similarity_knn, seen)

    # Use content filtering to find items similar to what the user has interacted with.
    content_items = content_filter(user_index, seen, mutable_items, item_features_knn)

    # If weighting is being applied to categories, multiply each categories weights by the weight_scale.
    if weight_category:
        for index, category in enumerate(weight_category):
            
            mutable_items[category] =  mutable_items[category] * weight_scale[index]
            mutable_users[category] += 0.5


    # Turn all items into a list.
    all_items = list(collaborative_items)+ list(content_items)

    # Filter items to remove those that the user has already seen.
    recommendable_items = filer_recommendations(all_items, seen, user_index)

    # Transform items into a set to sort them by index and then back into a list.
    items = list(set(recommendable_items))

    # Get the users feature row for calculating weights.
    user_row = mutable_users[mutable_users['user_id'] == ratings_df.iloc[user_index, 0]]

    # Calculate the predicted ratings' weights to include user preferences in recommendations.
    weights = calculate_weights(user_row, items, mutable_items)

    # Get the row of predicted ratings.
    ratings = R[1, items]
    ratings_overall = []

    # Iterate over all ratings multiplying each by its corresponding weight
    for i in range(len(weights)):
        ratings_overall.append(weights[i][0] * ratings[i])
    
    potential_recommendations = {items[i] : ratings_overall[i][0] for i in range(len(weights))}
    sorted_dict = dict(sorted(potential_recommendations.items(), key=lambda item: item[1], reverse=True))
    sorted_keys = list(sorted_dict.keys())
    return sorted_keys[:n]

def prepare_features(user_features, item_features, clustering, n_neighbors):
    """ 
    Prepares user and item features depending on the clustering type into a format needed for recommending items.
    Initially the user and item features are rearranged to be more efficient for the recommendation step and then 
    grouping of features is done depending on clustering. Once grouping of features has been done the features are placed in
    knn models. 
    
    Additionally applies feature grouping depending on the clustering
    """
    # Get the list of category and subcategory columns.
    cols = item_features.iloc[:, 4:-1].columns
    cols.sort_values()
    item_features = item_features.drop(columns=['title', 'abstract'])

    item_clusters = item_features['cluster'] 
    item_popularity = item_features['popularity']
    item_ids = item_features['news_id']


    user_clusters = user_features['cluster']
    user_median = user_features['median_time']
    user_ids = user_features['user_id']

    item_features = item_features[cols]
    user_features = user_features[cols]
    # [item_ids, item_clusters, item_popularity, item_features]
    # [user_ids, user_clusters, user_median, user_features]
    item_features = pd.concat([item_ids, item_clusters, item_popularity, item_features], axis=1)
    user_features = pd.concat([user_ids, user_clusters, user_median, user_features], axis=1)
    
    item_features_knn = NearestNeighbors(n_neighbors=n_neighbors, metric = 'euclidean')
    user_features_knn = NearestNeighbors(n_neighbors=n_neighbors, metric = 'euclidean')

    if clustering=="user":
        user_features = user_features.groupby('cluster').agg(sum).reset_index()

    if clustering=="item":
        cols = item_features.columns
        item_features = item_features.groupby('cluster').agg(sum).reset_index()
        item_features = item_features[cols]

    item_features_knn.fit(item_features.iloc[:, 1:])
    user_features_knn.fit(user_features.iloc[:, 1:])

    return user_features, item_features, user_features_knn, item_features_knn

def create_rec_data():

    users_df = pd.read_csv(data_path + "/user_features.csv", index_col=0)
    items_df = pd.read_csv(data_path + "/item_features.csv", index_col=0).drop(columns=[ "travel.1"])
    user_features, item_features = users_df.copy(), items_df.copy()
    user_features, item_features, user_features_knn, item_features_knn = prepare_features(user_features, item_features, 'item', 5)
    news = pd.read_csv(data_path + "/news.csv")

    U, V, item_idx, user_idx, full_ratings = get_predicted("item", 'gd', 25, 5)
    seen = {user_id : set(ratings) for user_id, ratings in user_idx.items()}
    R_hat = U @ V.T

    def get_items_ids(recommendations, item_features, items_df, news, clustering : bool = False):
        """
        Gets the feature representations of the items and a sample of item names if dealing with clusters.
        """
        if clustering == True:
            
            some_examples = {}
            for key in recommendations:
                ids = []
                # some_examples[f'Items from cluster number {key}'] = {"IDs" : [], "Titles" : []}
                cluster_top_4 = items_df[items_df['cluster'] == key].head(4)
                for id, title in zip(cluster_top_4['news_id'], cluster_top_4['title']):
                    ids.append(id)        
            return ids
        
        else:

            return news[news['news_id']] in item_features.iloc[recommendations, 1] 
        

    recommendable_counts = {}
    for i in range(2):
        for user_index in tqdm(range(500), total = 500): # first ten users
            if i == 0:
                recs = recommend_items(
                    R = R_hat,
                    seen = seen,
                    user_index = user_index,
                    ratings_df = full_ratings,
                    user_features = user_features,
                    item_features = item_features,
                    user_similarity_knn = user_features_knn,
                    item_features_knn = item_features_knn,
                    n = 5
                )
            else:
                recs = recommend_items(
                    R = R_hat,
                    seen = seen,
                    user_index = user_index,
                    ratings_df = full_ratings,
                    user_features = user_features,
                    item_features = item_features,
                    user_similarity_knn = user_features_knn,
                    item_features_knn = item_features_knn,
                    n = 5,
                    weight_category=['finance'],
                    weight_scale=[2]
                )

            ids = get_items_ids(recs, item_features, items_df, news, True)
            for id in ids:
                # print(news[news["news_id"] == id]['category'][0])
                category = news[news["news_id"] == id]['category'].to_list()[0] # row
                if category not in recommendable_counts:
                    recommendable_counts[category] = 1
                else:
                    recommendable_counts[category] += 1
            data = pd.DataFrame(recommendable_counts, index=[0]).melt()
            if i == 0:
                data.to_csv(data_path + "/predicted_items_unweighted.csv")
            else:
                data.to_csv(data_path + "/predicted_items_weighted.csv")
            
def plot_weight_results():
    """ 
    Plots counts of categories for 500 users recommendations' categories, once when no weights were applied and then
    once more with applied weights. 

    """

    weighted = pd.read_csv("../MIND_large/csv/predicted_items_weighted.csv", index_col=0)
    unweighted = pd.read_csv("../MIND_large/csv/predicted_items_unweighted.csv", index_col=0)
    weighted['weighted'] = [True for _ in range(13)]
    unweighted['weighted'] = [False for _ in range(14)]
    rec_counts = pd.concat([weighted, unweighted], axis=0)
    g = sns.catplot(data=rec_counts, kind="bar", col="weighted", x='value',y='variable', hue='variable', legend=False)
    g.set_titles(template="Finance Weighting: {col_name}")
    g.set_axis_labels(x_var = "Number of times recommended", y_var="Category");

def prepare_als_gd_output(model_output, als=True):
    """
    Prepares model output data for charting and visualizations.
    """
    model_output["RMSE"] = model_output["RMSE"].apply(eval)
    model_output["Max Updates"] = model_output["Max Updates"].apply(eval)
    columns = [i for i in range(len(model_output["RMSE"].iloc[1]))]
    rmses = pd.DataFrame()
    for index, row in model_output.iterrows():
        rmse = row['RMSE'] # gets the rmses
        data = {columns[i] : rmse[i] for i in range(len(rmse))}
        meep = pd.DataFrame(columns=columns, data=data, index=[index])
        rmses = pd.concat([rmses, meep])

    model_w_rmse = pd.concat([model_output, rmses], axis=1)
    if als:
        return model_w_rmse.drop(columns=["added_features", "RMSE", "Max Updates"]).melt(id_vars = ["clustering_type", "alg", "k", "lambda_reg"], var_name = "Iteration")
    else:
        return model_w_rmse.drop(columns=["added_features", "RMSE", "Max Updates"]).melt(id_vars = ["clustering_type", "alg", "k", "lambda_reg", "seed"], var_name = "Iteration")
def showcase_fm_results():
    fm_output = pd.read_csv('../MIND_large/csv/fm_testing_output.csv', index_col=0)
    fm_output['features'] = fm_output['features'].apply(eval).apply(lambda x : ", ".join(x))
    fm_output["RMSE"] = fm_output["RMSE"].apply(eval)
    fm_output["Max Updates"] = fm_output["Max Updates"].apply(eval)
    columns = [i for i in range(len(fm_output["RMSE"].iloc[1]))]
    rmses = pd.DataFrame()
    for index, row in fm_output.iterrows():
        rmse = row['RMSE'] # gets the rmses
        data = {columns[i] : rmse[i] for i in range(len(rmse))}
        meep = pd.DataFrame(columns=columns, data=data, index=[index])
        rmses = pd.concat([rmses, meep])
    model_w_rmse = pd.concat([fm_output, rmses], axis=1)
    model_w_rmse = model_w_rmse.drop(columns=['seed', 'alg', "RMSE", "Max Updates"]).melt(id_vars= ['features'], var_name = "Iteration")
    g = sns.FacetGrid(model_w_rmse, col="features", hue="features", height= 4)
    g.map(sns.lineplot, "Iteration", "value", marker="o", errorbar=None)
    g.add_legend(title="Regularization Level")
    # g.set_titles(template="Used Features: {col_name} ")
    g.set_axis_labels(x_var = "Iterations", y_var="RMSE")

def plot_model_output(als=True):
    """ 
    Plots resulting changes in rmse and updates functions over multiple parameter combinations. 

    Args:
    als: bool - changes what parameters are plotted if set to True or False. ALS was tested at different levels of k(1,5,25,50),
    lamba(1,5), and  gd was tested for different seed parameters.
    
    """

    if als:
        ALS_output = pd.read_csv("../MIND_large/csv/als_testing_output.csv", index_col=0)
        output = prepare_als_gd_output(ALS_output)
        g = sns.FacetGrid(output, col = 'k', row="clustering_type", hue="lambda_reg", aspect=1.5)
        g.map(sns.lineplot, "Iteration", "value", marker="o", errorbar=None)
        g.add_legend(title="Regularization Level")
        g.set_titles(template="Clustering type: {row_name}, k = {col_name} ")
        g.set_axis_labels(x_var = "Iterations", y_var="RMSE")
        plt.show()
    else:
        gd_output = pd.read_csv("../MIND_large/csv/gd_testing_output.csv", index_col=0)
        output = prepare_als_gd_output(gd_output, False)
        g = sns.FacetGrid(output, col = 'k', row="clustering_type", hue="seed", aspect=1.5)
        g.map(sns.lineplot, "Iteration", "value", marker="o", errorbar=None)
        g.add_legend(title="Random Seed")
        g.set_titles(template="Clustering type: {row_name}, k = {col_name} ")
        g.set_axis_labels(x_var = "Iterations", y_var="RMSE")
        plt.show()

def update_w_i(x_i_score, w_i, err, alpha):
    """
    Update an index in the feature vector w using the the score in the feature vector, its weight in w, the error and the learning rate alpha.

    Args:
        x_i_score (int) : The score in the feature vector x corresponding to the feature with weight w_i.
        w_i (int) : The weight of the feature in w.
        err (int) : The error calculated by finding y - y_hat.
        alpha (int) : The learning rate that was chosen at model creation. 

    Returns:
        w_i (int) : The updated model weight at the ith index in w.
    """
    return w_i + 2*alpha*err*x_i_score

def update_v_ij(x, v_ij, subset, row_i, err, alpha):
    """
    Updates the ith vector in the feature matrix V.

    Args:
        x (np.ndarray) : The row vector containing scores from a feature vector. 
        v_ij (np.ndarray) : A row vector from the feature matrix V. 
        subset (np.ndarray) : The subset of V corresponding to the indices of interacted with features.
        row_i (int) : The row index in the subset that is being updated.
        err (int) : The error calculated by finding y - y_hat.
        alpha (int) : The learning rate that was chosen at model creation. 
    
    Returns:
        Updated v_ij, the updated row vector in V.
    """
    # Get the number of rows in the subset to determine number of looping iterations.
    rows, k = subset.shape

    # Initialize a total to keep track of the sum.
    total = np.zeros((1, k))

    # Start looping over the rows of the subset not corresponding to the row determined by row_i.
    for j in range(rows):
        if j == row_i:
            pass
        else:
            total += subset[j, :] * x[j] - subset[row_i, :] * x[row_i]**2

    return v_ij + 2 * alpha * err * x[row_i] * total

def factorization_machine(feature_vectors, k, num_features, alpha):
    """
    Takes in a set of feature vectors, the desired number of latent factors and the learning rate and then
    performs gradient descent in the vain of a factorization machine to train model parameters w_0, w and V.

    Args:
        feature_vectors (dict) : The sparse representation of feature vectors as a dictionary with keys 'indices' and 'scores'
        k (int) : The desired number of latent factors. 
        num_features (int) : The maximum index in the row vectors which gets used in the creation of V.
        alpha (int) : The learning rate for the gradient descent.
    
    Returns:
        w_0, w and V which after running will be trained model weights that can be used on new observations.
    """

    # Initialize w_0.
    w_0 = 1
    
    # Initialize w.
    w = np.random.uniform(0, 1, size=num_features).reshape((1, num_features))
    w_old = np.zeros_like(w)

    # Initialize V
    V = np.random.uniform(0, 1, size=num_features*k).reshape((num_features, k))
    V_old = np.zeros_like(V)

    # Iterate through all rows provided by the feature feature vectors argument.
    for row in range(len(feature_vectors)):

        # Get the indices of the features that are used and scores.
        indices = feature_vectors[row]["indices"]
        scores = feature_vectors[row]["scores"]
        rating = feature_vectors[row]["rating"]
        # Subset V for the rows corresponding to rated feature indices.
        V_subset = V[indices, :]

        # Calculate y_hat.
        rating_estimate = y_hat(w_0, row, w, V_subset)
        error =  rating - rating_estimate

        for index in range(len(scores)):
            # We first update the ith weight in w.
            w[:, indices[index]] = update_w_i(scores[index], w[:, indices[index]], error, alpha)

            # We then update the ith row of the feature matrix V.
            V[indices[index], :] = update_v_ij(scores, V[indices[index], :], V_subset, index, error, alpha)

        w_0 = update_w_0(w_0, error, alpha)
    
    return w_0, w, V
