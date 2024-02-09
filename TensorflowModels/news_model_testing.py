import tensorflow as tf
import tensorflow_recommenders as tfrs
import keras.api._v2.keras as keras
import pandas as pd
import numpy as np
from typing import Dict, Text

"""
As modelling recommender systems with tensorflow is an intense task, will be starting with the basics to truly understand how it all fits together. Moving onto more complex models later on
"""

embedding_dimension = 32

def prepare_data():
    """
    Prepares the basic dataset with ratings and features for basic model initialization.
    """

    # What do we want to do here?
    # want to load in the data that has already had ratings assigned modeled after the movie lens dataset
    # then want to map it and such

news = pd.read_csv('fpath')

news = news.map(lambda x: x['news_title']) # <- will have to change and fix later on

unique_user_ids = None # <- fill in with prepared data

user_model = keras.Sequential([
keras.layers.StringLookup(vocabulary=unique_user_ids, mask_token=None),
keras.layers.Embedding(len(unique_user_ids) + 1, embedding_dimension)
])

unique_news_titles = None # <- fill in, do we want these to be the IDs? most likely, will have to check the movielens dataset to see what tutorial was doing

news_model = keras.Sequential([
  keras.layers.StringLookup(vocabulary=unique_news_titles, mask_token=None),
  keras.layers.Embedding(len(unique_news_titles) + 1, embedding_dimension)
])

# Set up metric for recommendations, for retrieval the factorized topK is used
metrics = tfrs.metrics.FactorizedTopK(
  candidates=news.batch(128).map(news_model)
)

# Set up loss metric
task = tfrs.tasks.Retrieval(
  metrics=metrics
)


class MIND_model(tfrs.Model):

    def __init__(self, user_model, news_model):
        super().__init__()
        self.news_model: keras.Model = news_model
        self.user_model: keras.Model = user_model
        self.task: keras.layers.Layer = task

    def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:

        # We pick out the user features and pass them into the user model giving us back embeddings.
        user_embeddings = self.user_model(features["user_id"])

        # Same for movies
        positive_news_embeddings = self.movie_model(features["news_title"])

        # The task computes the loss and the metrics.
        return self.task(user_embeddings, positive_news_embeddings)
    

# then from here we can train and evaluate our model.

