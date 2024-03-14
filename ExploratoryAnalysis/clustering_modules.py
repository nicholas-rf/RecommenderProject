import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')
import logging
logging.disable(logging.CRITICAL)

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import matplotlib.pyplot as plt 
import umap.plot as uplot
import umap.umap_ as umap
import hdbscan
from sklearn.preprocessing import LabelEncoder
import sklearn.cluster as cluster
import matplotlib
import plotly.express as px
from sklearn.impute import SimpleImputer
import distinctipy
import numpy as np
import os

"""
This module contains functions that create embeddings, create and apply clusters, and visualize clustering results.
"""

def vectorize_items(news_text):
    """ 
    Vectorizes the news dataset via its abstract and title under both TF-IDF and BOW vectorization methods.
    The result of the vectorization methods are two matrices containing the vectorized text.
    
    Args:
        news_text (pd.DataFrame) : The news dataset to vectorize.
    
    Returns:
        bow_matrix : The dataset vectorized into a BOW matrix.
        tf_matrix : The dataset vectorized into a TF-IDF matrix
    """

    # Fill in the NaN values with a " . " to avoid issues with vectorization.
    news_text['abstract'] = news_text['abstract'].fillna(' . ')

    # Initialize vectorizers from scikit-learn with english stop words.
    bow_vectorizer = CountVectorizer(stop_words='english')
    tf_vectorizer = TfidfVectorizer(stop_words='english')

    # Create bag of words and tf-idf matrices from all abstracts and titles.
    bow_matrix = bow_vectorizer.fit_transform(news_text['abstract'] + news_text['title'])
    tf_matrix = tf_vectorizer.fit_transform(news_text['abstract'] + news_text['title'])
    return bow_matrix, tf_matrix

def create_UMAP_embeddings(dimension, data, metric='hellinger',n_neighbors=30,min_dist=0.0,n_epochs=None):
    """
    Performes umap dimension reduction on the data up to the dimension specified by the associated parameter.

    Args:
        dimension (int) : The number of components to reduce the data to.
        data () : The data that is being reduced via umap.
        metric (str) : The distance metric that gets used to create embeddings.

    Returns:
        embeddings () : Returns dimension reduced embeddings via UMAP.
    """
    embeddings = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_epochs=n_epochs,
        n_components=dimension,
        metric = metric,
        random_state=42,
        n_jobs=1
    ).fit_transform(data)
    return embeddings

def create_hdbscan_labels(embeddings, min_samples=10, min_cluster_size=50):
    """ 
    Creates labels from the hdbscan clustering algorithm.
    
    Args:
        embeddings () : The dimension redued embeddings to make labels from.
        min_samples (int) : Determines the minimum samples required for a cluster under the hdbscan clustering algorithm.
        min_cluster_size (int) : Determines the minimum number of samples in a group for it to be considered as a cluster.

    Returns:
        labels () : Labels created by the HDBSCAN algorithm.
    """
    return hdbscan.HDBSCAN(min_samples=min_samples, min_cluster_size=min_cluster_size).fit_predict(embeddings)

def create_kmeans_labels(embeddings, n_clusters=10):
    """
    Creates labels from the kmeans clustering algorithm

    Args:
        embeddings () : The dimension redued embeddings to make labels from.
        n_clusters (int) : The number of clusters to create.

    Returns:
        labels () : Labels created by the HDBSCAN algorithm.
    """   
    return cluster.KMeans(n_clusters = n_clusters, n_init='auto', random_state=42).fit_predict(embeddings)

def visualize_all_item_clusters(bow_embeddings, tf_embeddings, distance_metric : str, hdbscan_labels : list, kmeans_labels : list, cmap : str = 'viridis'):
    """ 
    Visualizes clustering results of all vectorization methods and clustering algorithms using matplotlib.

    Args:
        bow_embeddings (list) A list of BOW embeddings under euclidean and hellinger.
        tf_embeddings (list) : A list of TF-IDF embeddings under euclidean and hellinger.
        distance_metric (list) : A list of distance metrics to use for plot titles.
        hdbscan_labels (list) : A list of hdbscan labels for the euclidean and hellinger distance metric BOW and TF-IDF embeddings.  
        kmeans_labels (list) : A list of kmeans labels for the euclidean and hellinger distance metric BOW and TF-IDF embeddings. 
        cmap (str) : The colormap to use within plotting.

    Returns:
        Creates and shows a chart of 2 rows and 4 columns containing scatter plots with clustering results.
    """
    # Create the grid to put plots on.
    fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(20,10))
    
    # Flatten the axes for easier list indexing.
    axs = axs.flatten()

    # Unpack the labels from the lists.
    bow_eu_hdb, tf_eu_hdb, bow_he_hdb, tf_he_hdb = hdbscan_labels
    bow_eu_km, tf_eu_km, bow_he_km, tf_he_km = kmeans_labels
    
    # Create euclidean distance metric scatterplots assigned to the top row.
    axs[0].scatter(bow_embeddings[0][:, 0],bow_embeddings[0][:, 1], alpha = 0.5, s=1, c=bow_eu_hdb, cmap=cmap)
    axs[1].scatter(bow_embeddings[0][:, 0],bow_embeddings[0][:, 1], alpha = 0.5, s=1, c=bow_eu_km, cmap=cmap)
    axs[2].scatter(tf_embeddings[0][:, 0], tf_embeddings[0][:, 1], alpha = 0.5, s=1, c=tf_eu_hdb, cmap=cmap)
    axs[3].scatter(tf_embeddings[0][:, 0], tf_embeddings[0][:, 1], alpha = 0.5, s=1, c=tf_eu_km, cmap=cmap)
    axs[0].set_title(f'{distance_metric[0]}: BoW Embeddings - hdbscan')
    axs[1].set_title(f'{distance_metric[0]}: BoW Embeddings - kmeans')
    axs[2].set_title(f'{distance_metric[0]}: TF-IDF Embeddings - hdbscan')
    axs[3].set_title(f'{distance_metric[0]}: TF-IDF Embeddings - kmeans')
    
    # Create hellinger distance metric scatterplots assigned to the bottom row.
    axs[4].scatter(bow_embeddings[1][:, 0],bow_embeddings[1][:, 1], alpha = 0.5, s=1, c=bow_he_hdb, cmap=cmap)
    axs[5].scatter(bow_embeddings[1][:, 0],bow_embeddings[1][:, 1], alpha = 0.5, s=1, c=bow_he_km, cmap=cmap)
    axs[6].scatter(tf_embeddings[1][:, 0], tf_embeddings[1][:, 1], alpha = 0.5, s=1, c=tf_he_hdb, cmap=cmap)
    axs[7].scatter(tf_embeddings[1][:, 0], tf_embeddings[1][:, 1], alpha = 0.5, s=1, c=tf_he_km, cmap=cmap)
    axs[4].set_title(f'{distance_metric[1]}: BoW Embeddings - hdbscan')
    axs[5].set_title(f'{distance_metric[1]}: BoW Embeddings - kmeans')
    axs[6].set_title(f'{distance_metric[1]}: TF-IDF Embeddings - hdbscan')
    axs[7].set_title(f'{distance_metric[1]}: TF-IDF Embeddings - kmeans')
    
    # Assign the plot a tight layout.
    plt.tight_layout()

def item_cluster_exploration():
    """
    Combines several clustering functions and the visualization function together to make parameter exploration for item clustering more efficient
    and to improve readability of the clustering report.  
    """
    # Loading in the data for tf-idf and bag of words vectorization methods.
    news_text = pd.read_csv('../MIND_large/csv/news.csv', index_col=0).set_index('news_id').drop(columns=['url','title_entities','abstract_entities'])

    # Create our UMAP_embeddings for our vectorization types and distance metrics.
    bow_matrix, tf_matrix = vectorize_items(news_text)
    bow_embeddings = [create_UMAP_embeddings(2, bow_matrix, 'euclidean'), create_UMAP_embeddings(2, bow_matrix)]
    tf_embeddings = [create_UMAP_embeddings(2, tf_matrix, 'euclidean'), create_UMAP_embeddings(2, tf_matrix)]
    
    # Apply kmeans and hdbscan clustering algorithms to our embeddings
    embeddings = bow_embeddings + tf_embeddings

    kmeans_labels = [create_kmeans_labels(SimpleImputer(strategy='mean').fit_transform(embeddings[index]), n_clusters=30) for index in [0, 2, 1, 3]]
    hdbscan_labels = [create_hdbscan_labels(embeddings[index]) for index in [0, 2, 1, 3]]

    # Plot clustering results
    colors = distinctipy.get_colors(30)
    cmap = distinctipy.get_colormap(colors)
    try:
        visualize_all_item_clusters(bow_embeddings, tf_embeddings, ['Euclidean', 'Hellinger'], hdbscan_labels, kmeans_labels, cmap=cmap)   
    except:
        print("CMAP did not work")
        visualize_all_item_clusters(bow_embeddings, tf_embeddings, ['Euclidean', 'Hellinger'], hdbscan_labels, kmeans_labels, cmap=cmap)   

def item_cluster(item_features, n_clusters, metric='hellinger', matrix_type='tf-idf', cluster_type = 'kmeans'):
    """
    Appends n_clusters clusters to the item features dataset with kmeans. Either will load in item embeddings or make them if not found.
    """
    # Standard filepath for the item embeddings.
    fpath = "../MIND_large/embeddings/item_embeddings.npy"

    # Read in the news and its text, then create the tf-idf matrix.
    news_text = pd.read_csv('../MIND_large/csv/news.csv', index_col=0).set_index('news_id').drop(columns=['url','title_entities','abstract_entities'])
    bow_matrix, tf_matrix = vectorize_items(news_text)

    # Delete the dataset to save memory.
    del news_text

    # If the file for embeddings does not exist yet, create the embeddings and save them.
    if not os.path.exists(fpath):
        print("Item embeddings not found, creating now")
        if matrix_type == 'tf-idf':
            print("TF-IDF matrix type selected")
            embeddings = create_UMAP_embeddings(2, tf_matrix, metric=metric)
            np.save(fpath, embeddings)
        elif matrix_type == 'bow':
            embeddings = create_UMAP_embeddings(2, bow_matrix, metric=metric)
            np.save(fpath, embeddings)                

    # Otherwise load the embeddings.
    else:
        print("Item embeddings found, loading now")
        embeddings = np.load(fpath)

    # Create kmeans labels with the embeddings for the number of clusters specified in the arguments of item_cluster
    if cluster_type == 'kmeans':
        labels = create_kmeans_labels(SimpleImputer(strategy='mean').fit_transform(embeddings), n_clusters)
    elif cluster_type == 'hdbscan':
        labels = create_hdbscan_labels(SimpleImputer(strategy='mean').fit_transform(embeddings), n_clusters)

    # Apply the new cluster labels and return the item features.
    item_features['cluster'] = labels
    return item_features

def user_cluster_exploration():
    """
    Minimizes boilerplate within clustering report to improve readability. Combines all methods necessary to create
    and then visualize exploratory user clustering results.
    """
    # Create lists of experimented on parameters to iterate through.
    metrics = ['euclidean','cosine']
    parameters = [(0.0, 30),(0.0, 50),(0.1, 30),(0.1,50)]
    embeddings = []

    # Iterate over parameters loading in the related embeddings.
    for metric in metrics:
        for p_comb in parameters:
            min_dist, n_neigh = p_comb
            embeddings.append(np.load(f'../MIND_large/embeddings/user_embeddings_{metric}_{min_dist}_{n_neigh}.npy'))
        
    # Create labels for each embedding and then plot the results.
    kmeans_labels = [cluster.KMeans(n_clusters=50,n_init='auto').fit_predict(embedding) for embedding in embeddings]
    plot_user_clusters_params(embeddings, kmeans_labels, metrics, parameters)

def plot_user_clusters_params(embeddings, kmeans, distance_metric, params):
    """
    Plots the results of exploration on umap parameters for user clustering.

    Args:
        embeddings () : 
        kmeans () :
        distance_metric () : 
        params () :
    
    Returns:
        ... 
    """
    colors = distinctipy.get_colors(50)
    cmap = distinctipy.get_colormap(colors)

    # Create the grid to put plots on.
    fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(20,10))
    
    # Flatten the axes for easier list indexing.
    axs = axs.flatten()

    # Unpack the labels from the lists.
    eu_030_km, eu_050_km, eu_130_km, eu_150_km, cos_030_km, cos_050_km, cos_130_km, cos_150_km = kmeans
    
    # Create euclidean distance metric scatterplots assigned to the top row.
    axs[0].scatter(embeddings[0][:, 0],embeddings[0][:, 1], alpha = 0.5, s=1, c=eu_030_km, cmap=cmap)
    axs[1].scatter(embeddings[1][:, 0],embeddings[1][:, 1], alpha = 0.5, s=1, c=eu_050_km, cmap=cmap)
    axs[2].scatter(embeddings[2][:, 0],embeddings[2][:, 1], alpha = 0.5, s=1, c=eu_130_km, cmap=cmap)
    axs[3].scatter(embeddings[3][:, 0],embeddings[3][:, 1], alpha = 0.5, s=1, c=eu_150_km, cmap=cmap)

    axs[0].set_title(f'{distance_metric[0]}-{" ".join(map(str, params[0]))}')
    axs[1].set_title(f'{distance_metric[0]}-{" ".join(map(str, params[1]))}')
    axs[2].set_title(f'{distance_metric[0]}-{" ".join(map(str, params[2]))}')
    axs[3].set_title(f'{distance_metric[0]}-{" ".join(map(str, params[3]))}')

    axs[4].scatter(embeddings[3][:, 0],embeddings[3][:, 1], alpha = 0.5, s=1, c=cos_030_km, cmap=cmap)
    axs[5].scatter(embeddings[4][:, 0],embeddings[4][:, 1], alpha = 0.5, s=1, c=cos_050_km, cmap=cmap)
    axs[6].scatter(embeddings[5][:, 0],embeddings[5][:, 1], alpha = 0.5, s=1, c=cos_130_km, cmap=cmap)
    axs[7].scatter(embeddings[6][:, 0],embeddings[6][:, 1], alpha = 0.5, s=1, c=cos_150_km, cmap=cmap)

    axs[4].set_title(f'{distance_metric[1]}-{" ".join(map(str, params[0]))}')
    axs[5].set_title(f'{distance_metric[1]}-{" ".join(map(str, params[1]))}')
    axs[6].set_title(f'{distance_metric[1]}-{" ".join(map(str, params[2]))}')
    axs[7].set_title(f'{distance_metric[1]}-{" ".join(map(str, params[3]))}')

def user_cluster(features, n_clusters=50, metric='euclidean', min_dist=0.1,n_neigh=50):
    """
    Applies clustering and returns new features dataframe with specified KNN clusters. Can also create UMAP embeddings to apply to 
    features if they are not already present in the users MIND_large directory. All applicable UMAP parameters are also availible as 
    parameters.
    """

    # Initialize the path that currently points to or will point to user embeddings.
    path = "../MIND_large/embeddings/user_embeddings.npy"

    # If the path does not exist yet, make the embeddings and save them. Otherwise load them.
    if not os.path.exists(path):
        print("User embeddings not found, creating now")
        embeddings = create_UMAP_embeddings(2, features.iloc[:,1:], metric, min_dist=min_dist, n_neigh=n_neigh)
        np.save(path, embeddings)
    else:
        print("User embeddings found, loading now")
        embeddings = np.load(path)
    
    # Create the kmeans labels and append them to the features.
    labels = create_kmeans_labels(embeddings, n_clusters=n_clusters)
    features_new = features.copy()
    features_new["cluster"] = labels

    # Return the features.
    return features_new

def chart(trans):
    #--------------------------------------------------------------------------#
    # This section is not mandatory as its purpose is to sort the data by label 
    # so, we can maintain consistent colors for digits across multiple graphs
    
    # Create a Pandas dataframe using the above array
    df=pd.DataFrame(trans, columns=['x', 'y', 'z'])
    #--------------------------------------------------------------------------#
    
    # Create a 3D graph
    fig = px.scatter_3d(df, x='x', y='y', z='z', height=900, width=950)

    # Update chart looks
    fig.update_layout(title_text='UMAP',
                      showlegend=True,
                      legend=dict(orientation="h", yanchor="top", y=0, xanchor="center", x=0.5),
                      scene_camera=dict(up=dict(x=0, y=0, z=1), 
                                            center=dict(x=0, y=0, z=-0.1),
                                            eye=dict(x=1.5, y=-1.4, z=0.5)),
                                            margin=dict(l=0, r=0, b=0, t=0),
                      scene = dict(xaxis=dict(backgroundcolor='white',
                                              color='black',
                                              gridcolor='#f0f0f0',
                                              title_font=dict(size=10),
                                              tickfont=dict(size=10),
                                             ),
                                   yaxis=dict(backgroundcolor='white',
                                              color='black',
                                              gridcolor='#f0f0f0',
                                              title_font=dict(size=10),
                                              tickfont=dict(size=10),
                                              ),
                                   zaxis=dict(backgroundcolor='lightgrey',
                                              color='black', 
                                              gridcolor='#f0f0f0',
                                              title_font=dict(size=10),
                                              tickfont=dict(size=10),
                                             )))
    # Update marker size
    fig.update_traces(marker=dict(size=3, line=dict(color='black', width=0.1)))
    
    fig.show()



# might want to consider making user features with sub category as well and have it match the item features? 
# that way we could inform the weights by sub-category and category and more, yes I think thats good