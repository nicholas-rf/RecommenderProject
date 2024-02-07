DESIGN DOCUMENT FOR TENSORFLOW MODELS:

## Tensorflow model files

tensorflow_recommenders_quickstart.py : Quickstart implementation for a tensorflow matrix factorization model with the movielens dataset. More information found within project file. 

tensorflow_retrieval_docs.py : A tensorflow recommenders model for retrieval utilizing the movielens dataset and no additional features. Includes notes on whats happening for comprehension. 

tensorflow_ranking_docs.py : A tensorflow recommenders model for ranking utilizing the movielens dataset and no additional features. Includes notes on whats happening for comprehension. Ranking is the next step in a recommender after retrieval and helps fine-tune recommendations for the model.

tensorflow_sequential_docs.py : A tensorflow recommenders model that changes the query / user tower in the recommender system to one that has a GRU layer in the keras sequential model so that a recurrent neural network can predict the users next interacted item.
