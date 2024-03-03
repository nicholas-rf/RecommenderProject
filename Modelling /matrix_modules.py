import pandas as pd
import numpy as np

def load_dataset(train_split = '80_20'):
    """
    Loads in the full training dataset predicated upon the train test split specified. 
    """
    full = pd.DataFrame()
    for i in range(2):
        df = pd.read_csv(f"../MIND_large/{train_split}/train_chunk{i}.csv", index_col=0)
        full = pd.concat([full, df])   
    news = pd.read_csv('../MIND_large/csv/news_cluster_labels.csv')
    all_ratings = full.groupby('user_id')['news_id'].apply(list).reset_index()
    scores = full.groupby('user_id')['score'].apply(list).reset_index()
    all_ratings['scores'] = scores['score']
    user_clustered = pd.read_csv('../MIND_large/csv/full_user_clusters.csv') # needs to be updated for the train test split as well :D 
    return all_ratings, news, user_clustered

def create_item_cluster_mat(ratings_df, news, num_users = 255990, num_clusters = 30, isALS=False):
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
        
    """
    # Initialize the hash map that will create the matrix as a list of np zero arrays for each cluster and the hashmap of item clusters .
    item_clusters = {item : cluster for item, cluster in zip(news['news_id'], news['labels'])}
    matrix = {cluster : np.full(num_users, 0, dtype='int8') for cluster in range(num_clusters)} 
    
    if isALS:
        # Initialize the cluster hashmap, which is used to track an item index and all row indices that engage with that item.
        cluster_idx = {cluster : set() for cluster in range(num_clusters)}

        # Initialize the user hashmap, which is used to track a user index and all item indices that the user engaged with.
        user_idx = {user_id : set() for user_id in range(num_users)}
        
    # Initialize a counter to keep track of user index.
    counter = 0

    # Iterate over every user, their ratings and score in the rating matrix .
    for user, ratings, score in zip(ratings_df['user_id'], ratings_df['news_id'], ratings_df['scores']):
        for index in range(len(ratings)):
            # Get the news id of the interaction and their rating.
            news_id = ratings[index]
            num = score[index]

            # If the rating is not zero, add a 1 to the cluster of the article.
            if num != 0:    
                matrix[item_clusters[news_id]][counter] += 1
                
                # If we are using ALS, add relevant indices to the hashmaps.
                if isALS:
                    cluster_idx[item_clusters[news_id]].add(counter)
                    user_idx[counter].add(item_clusters[news_id])
        counter += 1
    
    # Return the full matrix and arrays if we are using ALS, otherwise return only the matrix
    return np.column_stack(list(matrix.values())), cluster_idx, user_idx if isALS else np.column_stack(list(matrix.values()))

def create_user_cluster_mat(ratings_df, news, user_clustered, num_user_clusters=10, isALS=False):
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
    """
    # Initialize the hash map that will create the matrix as a list of np zero arrays for each news_id and the hashmap of user clusters.
    matrix = {news_id : np.full(num_user_clusters, 0, dtype='int8') for news_id in news['news_id']}
    user_clusters = {user : cluster for user, cluster in zip(user_clustered['user_id'], user_clustered['cluster'])}
    
    if isALS:
        # Initialize the item idx hash map to create a hash map that can be used to check all row indices that have an appearance in the column.
        item_lookup = {news_id : index for index, news_id in enumerate(news['news_id'])}
        item_idx = {index : set() for index in range(len(news['news_id']))}

        # Initialize the cluster idx hash map to create a hash map that can be used to check all column indicies that have an appearance in the row.
        cluster_idx = {cluster : set() for cluster in range(num_user_clusters)}

    
    # Iterate over the user ids, ratings and scores.
    for user, ratings, score in zip(ratings_df['user_id'], ratings_df['news_id'], ratings_df['scores']):
        
        # Determine the users cluster and then iterate all of their ratings
        cluster = user_clusters[user]
        for index in range(len(ratings)):
            # Get the news ID and score of their rating.
            news_id = ratings[index]
            num = score[index]

            # If the score is not zero, add 1 to the clusters score for that item
            if num != 0:
                matrix[news_id][cluster] += 1  # Access the column of the matrix, and then find the index in that column for that users cluster, then increment by 1
                
                # If we are using ALS, add the relevant indices to the hash map.
                if isALS:
                    # Add the corresponding cluster number to the column index for the news id.
                    item_idx[item_lookup[news_id]].add(cluster)

                    # Add the column index to the clusters key.
                    cluster_idx[cluster].add(item_lookup[news_id])
        
    return np.column_stack(list(matrix.values())), item_idx,  cluster_idx if isALS else np.column_stack(list(matrix.values()))

