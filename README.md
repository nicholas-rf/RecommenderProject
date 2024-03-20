# Recommender System Project
## Project outline
This serves as the final project for PSTAT 134/234 by Coby Wilcox and Nicholas Roze-Freitas. Our goal was to utilize the Microsoft News Dataset, MIND, to evaluate the efficacy of different recommender system models frameworks ranging from state of the art to classic. The main report can be read within the main directory in addition to supplementary materials: eda_report.ipynb, clustering_report.ipynb, model_report.ipynb, tensorflow_report.ipynb and data_processing_documentation.ipynb. Descriptions of each reports contents can be found immediately below.


## Directory Description

* clustering_report.ipynb : Discusses results regarding exploration of UMAP dimension reduction and clustering.
* data_processing_documentation.ipynb : Discusses and walks through data processing.
* eda_report.ipynb : Exploratory data analysis findings and discussion.
* model_report.ipynb : A more in depth look at modelling practices and mathematics behind what we did.


* .devcontainer
    * Contains the docker image and devcontainer files we used as our environment for this project.
* Exploratory Analysis : Files related to data processing, clustering and EDA.
    * clustering_modules.py : Contains all functions that are used for clustering ranging from exploration of parameters to application of different amounts of clusters.
    * data_processing_modules.py : Contains all functions that are used for processing and transforming our data.
    * user_cluster_script.py : A script that is used to generate the dimension reduced embeddings used for exploration of user clustering parameters.
    * visualization_modules.py : Contains several functions used to generate visualizations for exploratory data analysis.
* MIND_large : Data
    * CSV
        * als_testing_output.csv : ALS parameter testing outputs of RMSE and maximum update.
        * behaviors.csv : The user behaviors data.
        * category_with_popularity.csv : The categories and popularity counts.
        * item_features.csv : The dataset of items and their features.
        * news_with_popularity.csv : The news dataset with popularity counts attached to each article.
        * news.csv : The news dataset detailing the catalog for the recommender system. 
        * gd_testing_output.csv : The gradient descent parameter testing outputs of RMSE and maximum updates.
        * tensorflow_dataset_chunk{i}.csv : The ith chunk of the full Tensorflow dataset.
        * user_features.csv : The dataset of users and their features.

    * Embeddings : Contains umap reduced embeddings as well as factorization machine sparse matrix files.

* Modelling : Files related to matrix factorization modelling.
    * fastFM : The factorization machine library.
    * matrix_modules.py : 


## Environment Instructions
A docker container containing an anaconda python3.11 virtual environment was utilized for this project. The required docker container can be found in the repository in addition to the devcontainer. 

## Hardware Specifications
The computer that was used to run a majority of this project was a virtual machine with 60gb of RAM, a 16 core CPU and a 60gb root disk.

## Special thanks
We extend a special thank you to Professor Oh and the NSF for providing computational resources that allowed us to explore this project fully. 
