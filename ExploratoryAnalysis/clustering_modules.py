import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import matplotlib.pyplot as plt 
import umap.plot as uplot
import umap.umap_ as umap
import hdbscan
from sklearn.preprocessing import LabelEncoder
import sklearn.cluster as cluster
import matplotlib
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def vectorize_items(news_text):
    """ 
    Vectorizes the news dataset via its abstract and title under both TF-IDF and BOW vectorization methods.
    
    Args:
        news_text (pd.DataFrame) : The news dataset to vectorize.
    
    Returns:
        bow_matrix : The dataset vectorized into a BOW matrix.
        tf_matrix : The dataset vectorized into a TF-IDF matrix
    """

    news_text['abstract'] = news_text['abstract'].fillna(' . ')

    # Initialize vectorizers from scikit-learn
    bow_vectorizer = CountVectorizer(stop_words='english')
    tf_vectorizer = TfidfVectorizer(stop_words='english')

    # Create bag of words and tf-idf matrices of the corpus
    bow_matrix = bow_vectorizer.fit_transform(news_text['abstract'] + news_text['title'])
    tf_matrix = tf_vectorizer.fit_transform(news_text['abstract'] + news_text['title'])
    return bow_matrix, tf_matrix


def create_UMAP_embeddings(dimension, data, metric='hellinger',n_neighors=30,min_dist=0.0):
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
        n_neighbors=30,
        min_dist=0.0,
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
    return cluster.KMeans(n_clusters = n_clusters, n_init='auto').fit_predict(embeddings)
    
def visualize_item_clusters(bow_embeddings, tf_embeddings, distance_metric : str, hdbscan_labels : list, kmeans_labels : list, cmap : str = 'plasma'):
    """ 
    Visualizes clustering results using matplotlib.
    """
    cmap = matplotlib.colormaps[cmap]
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(15,10))
    # labels = [bow_hellinger_hdbscan_labels, bow_hellinger_kmeans_labels, tf_hellinger_hdbscan_labels, tf_hellinger_kmeans_labels]
    axs = axs.flatten()
    bow_hdb, tf_hdb = hdbscan_labels
    bow_km, tf_km = kmeans_labels
    axs[0].scatter(bow_embeddings[:, 0],bow_embeddings[:, 1], alpha = 0.5, s=1, c=bow_hdb, cmap=cmap)
    axs[1].scatter(bow_embeddings[:, 0],bow_embeddings[:, 1], alpha = 0.5, s=1, c=bow_km, cmap=cmap)
    axs[2].scatter(tf_embeddings[:, 0], tf_embeddings[:, 1], alpha = 0.5, s=1, c=tf_hdb, cmap=cmap)
    axs[3].scatter(tf_embeddings[:, 0], tf_embeddings[:, 1], alpha = 0.5, s=1, c=tf_km, cmap=cmap)
    fig.suptitle(f"{distance_metric} Distance Metric")
    axs[0].set_title('BoW Embeddings - hdbscan')
    axs[1].set_title('BoW Embeddings - kmeans')
    axs[2].set_title('TF-IDF Embeddings - hdbscan')
    axs[3].set_title('TF-IDF Embeddings - kmeans')
    plt.tight_layout()

def visualize_all_item_clusters(bow_embeddings, tf_embeddings, distance_metric : str, hdbscan_labels : list, kmeans_labels : list, cmap : str = 'plasma'):
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
    # Initialize the colormap.
    cmap = matplotlib.colormaps[cmap]

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

