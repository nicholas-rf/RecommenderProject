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

# We need to first start by casting our dataset into a tf.data.Datasets object so that it is compatible with our tensorflow model.
user_data = pd.read_csv('fpath')
news_catalog = pd.read_csv('fpath')

# With our data loaded in we can then place them into a tensorflow dataset.
user_data = tf.data.Dataset.from_tensor_slices(dict(user_data))
news_catalog = tf.data.Dataset.from_tensor_slices(dict(news_catalog))

# Then with our finalized data we can move on to mapping it into a dataset object.
ratings = user_data.map(lambda x: {
    'user_id': x['user_id'],
    'news_id': x['news_id'],
    'score': x['score']
})

news = news_catalog.map(lambda x: x["news_id"]) 

# Then with our finalized data we can move on to initializing a model. 
tf.random.set_seed(42)
shuffled = ratings.shuffle(100_000, seed=42, reshuffle_each_iteration=False)

train = shuffled.take(80_000)
test = shuffled.skip(80_000).take(20_000)

news_titles = news.batch(1_000)
user_ids = ratings.batch(1_000_000).map(lambda x: x["user_id"])

unique_news_titles = np.unique(np.concatenate(list(news_titles)))
unique_user_ids = np.unique(np.concatenate(list(user_ids)))

## Layer definition within the model.

# user_model = keras.Sequential([
#   keras.layers.StringLookup(
#       vocabulary=unique_user_ids, mask_token=None),
#   # We add 1 to account for the unknown token.
#   keras.layers.Embedding(len(unique_user_ids) + 1, embedding_dimension)
# ])

# news_model = keras.Sequential([
#   keras.layers.StringLookup(vocabulary=unique_news_titles, mask_token=None),
#   keras.layers.Embedding(len(unique_news_titles) + 1, embedding_dimension)
# ])

# tfrs.tasks.Ranking(
#     loss=keras.losses.MeanSquaredError(),
#     metrics=[keras.metrics.RootMeanSquaredError()],
# )

# tfrs.tasks.Retrieval(
#     metrics=tfrs.metrics.FactorizedTopK(
#         candidates=news.batch(128)
#     )
# )

class MindModel(tfrs.models.Model):
  def __init__(self, rating_weight: float, retrieval_weight: float) -> None:
    # We take the loss weights in the constructor: this allows us to instantiate
    # several model objects with different loss weights.

    super().__init__()

    embedding_dimension = 32

    # User and movie models.
    self.news_model: keras.layers.Layer = keras.Sequential([
      keras.layers.StringLookup(vocabulary=unique_news_titles, mask_token=None),
      keras.layers.Embedding(len(unique_news_titles) + 1, embedding_dimension)
    ])
    self.user_model: keras.layers.Layer = keras.Sequential([
      keras.layers.StringLookup(
          vocabulary=unique_user_ids, mask_token=None),
      # We add 1 to account for the unknown token.
      keras.layers.Embedding(len(unique_user_ids) + 1, embedding_dimension)
    ])

    # A small model to take in user and movie embeddings and predict ratings.
    # We can make this as complicated as we want as long as we output a scalar
    # as our prediction.
    self.rating_model = tf.keras.Sequential([
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(1),
    ])

    # The tasks.
    self.rating_task: keras.layers.Layer = tfrs.tasks.Ranking(
        loss=keras.losses.MeanSquaredError(),
        metrics=[keras.metrics.RootMeanSquaredError()],
    )
    self.retrieval_task: tf.keras.layers.Layer = tfrs.tasks.Retrieval(
        metrics=tfrs.metrics.FactorizedTopK(
            candidates=news.batch(128).map(self.movie_model)
        )
    )

    # The loss weights.
    self.rating_weight = rating_weight
    self.retrieval_weight = retrieval_weight

  def call(self, features: Dict[Text, tf.Tensor]) -> tf.Tensor:
    # We pick out the user features and pass them into the user model.
    user_embeddings = self.user_model(features["user_id"])
    # And pick out the movie features and pass them into the movie model.
    movie_embeddings = self.movie_model(features["news_id"])

    return (
        user_embeddings,
        movie_embeddings,
        # We apply the multi-layered rating model to a concatentation of
        # user and movie embeddings.
        self.rating_model(
            tf.concat([user_embeddings, movie_embeddings], axis=1)
        ),
    )

  def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:

    ratings = features.pop("score")

    user_embeddings, movie_embeddings, rating_predictions = self(features)

    # We compute the loss for each task.
    rating_loss = self.rating_task(
        labels=ratings,
        predictions=rating_predictions,
    )
    retrieval_loss = self.retrieval_task(user_embeddings, movie_embeddings)

    # And combine them using the loss weights.
    return (self.rating_weight * rating_loss
            + self.retrieval_weight * retrieval_loss)
  

    


