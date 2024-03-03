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

def item_cluster():
    """
    Minimizes boilerplate within clustering report to improve readability. Combines all methods necessary to create
    and then visualize exploratory item clustering results.
    """
    import warnings
    warnings.filterwarnings('ignore')

    if not os.path.exists('../MIND_large/csv/item_cluster0.npy'):
        # Loading in the data for tf-idf and bag of words vectorization methods.
        news_text = pd.read_csv('../MIND_large/csv/news.csv', index_col=0).set_index('news_id').drop(columns=['url','title_entities','abstract_entities'])

        # Create our UMAP_embeddings for our vectorization types and distance metrics.
        bow_matrix, tf_matrix = vectorize_items(news_text)
        bow_embeddings = [create_UMAP_embeddings(2, bow_matrix, 'euclidean'), create_UMAP_embeddings(2, bow_matrix)]
        tf_embeddings = [create_UMAP_embeddings(2, tf_matrix, 'euclidean'), create_UMAP_embeddings(2, tf_matrix)]
        
        # Apply kmeans and hdbscan clustering algorithms to our embeddings
        embeddings = bow_embeddings + tf_embeddings
        for num, embedding in enumerate(embeddings):
            np.save(f'../MIND_large/csv/item_cluster_{num}.npy', embedding)
        kmeans_labels = [create_kmeans_labels(SimpleImputer(strategy='mean').fit_transform(embeddings[index]), n_clusters=30) for index in [0, 2, 1, 3]]
        hdbscan_labels = [create_hdbscan_labels(embeddings[index]) for index in [0, 2, 1, 3]]

        # Plot clustering results
        colors = distinctipy.get_colors(50)
        visualize_all_item_clusters(bow_embeddings, tf_embeddings, ['Euclidean', 'Hellinger'], hdbscan_labels, kmeans_labels, cmap='viridis')   

        cluster_labels = kmeans_labels[3]
        news_text['labels'] = cluster_labels
        news_text.to_csv('../MIND_large/csv/clustered_items.csv')

    else:
        pass
    
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


### Parameter explanation for UMAP
# reducer = umap.UMAP(
#                n_neighbors=100, # default 15, The size of local neighborhood (in terms of number of neighboring sample points) used for manifold approximation.
#                n_components=3, # default 2, The dimension of the space to embed into.
#                metric='euclidean', # default 'euclidean', The metric to use to compute distances in high dimensional space.
#                n_epochs=1000, # default None, The number of training epochs to be used in optimizing the low dimensional embedding. Larger values result in more accurate embeddings. 
#                learning_rate=1.0, # default 1.0, The initial learning rate for the embedding optimization.
#                init='spectral', # default 'spectral', How to initialize the low dimensional embedding. Options are: {'spectral', 'random', A numpy array of initial embedding positions}.
#                min_dist=0.1, # default 0.1, The effective minimum distance between embedded points.
#                spread=1.0, # default 1.0, The effective scale of embedded points. In combination with ``min_dist`` this determines how clustered/clumped the embedded points are.
#                low_memory=False, # default False, For some datasets the nearest neighbor computation can consume a lot of memory. If you find that UMAP is failing due to memory constraints consider setting this option to True.
#                set_op_mix_ratio=1.0, # default 1.0, The value of this parameter should be between 0.0 and 1.0; a value of 1.0 will use a pure fuzzy union, while 0.0 will use a pure fuzzy intersection.
#                local_connectivity=1, # default 1, The local connectivity required -- i.e. the number of nearest neighbors that should be assumed to be connected at a local level.
#                repulsion_strength=1.0, # default 1.0, Weighting applied to negative samples in low dimensional embedding optimization.
#                negative_sample_rate=5, # default 5, Increasing this value will result in greater repulsive force being applied, greater optimization cost, but slightly more accuracy.
#                transform_queue_size=4.0, # default 4.0, Larger values will result in slower performance but more accurate nearest neighbor evaluation.
#                a=None, # default None, More specific parameters controlling the embedding. If None these values are set automatically as determined by ``min_dist`` and ``spread``.
#                b=None, # default None, More specific parameters controlling the embedding. If None these values are set automatically as determined by ``min_dist`` and ``spread``.
#                random_state=42, # default: None, If int, random_state is the seed used by the random number generator;
#                metric_kwds=None, # default None) Arguments to pass on to the metric, such as the ``p`` value for Minkowski distance.
#                angular_rp_forest=False, # default False, Whether to use an angular random projection forest to initialise the approximate nearest neighbor search.
#                target_n_neighbors=-1, # default -1, The number of nearest neighbors to use to construct the target simplcial set. If set to -1 use the ``n_neighbors`` value.
#                #target_metric='categorical', # default 'categorical', The metric used to measure distance for a target array is using supervised dimension reduction. By default this is 'categorical' which will measure distance in terms of whether categories match or are different. 
#                #target_metric_kwds=None, # dict, default None, Keyword argument to pass to the target metric when performing supervised dimension reduction. If None then no arguments are passed on.
#                #target_weight=0.5, # default 0.5, weighting factor between data topology and target topology.
#                transform_seed=42, # default 42, Random seed used for the stochastic aspects of the transform operation.
#                verbose=False, # default False, Controls verbosity of logging.
#                unique=False, # default False, Controls if the rows of your data should be uniqued before being embedded. 
#               )
def plot_user_clusters_params(embeddings, kmeans, distance_metric, params):
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