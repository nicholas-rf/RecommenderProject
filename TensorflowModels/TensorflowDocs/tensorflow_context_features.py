### Many factors affect whether features beyond ids are useful in a recommender model


# importance of context
# if preferences are contextual adding context features improves model drastically
# certain items may be popular only upon release at that point in time, but others might be 
# an evergreen meaning they are enjoyed for a long period of time
# query timestamps can play an important role in modelling popularity dynamics

# data sparsity
# non-id features can be critical if data is sparse
# if there are only a few observations available for a given user or item, the model may struggle with estimating a good per user per item representaiton
# other features such as item description, category and images are used to help the model generalize well past training stage
# relevant in the cold-start situations

# Preliminary installations

import os
import tempfile
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_recommenders as tfrs
import keras.api._v2.keras as keras

# follow the feature processing tutorial for userID timestamp and movie title features
ratings = tfds.load("movielens/100k-ratings", split="train")
movies = tfds.load("movielens/100k-movies", split="train")

ratings = ratings.map(lambda x: {
    "movie_title": x["movie_title"],
    "user_id": x["user_id"],
    "timestamp": x["timestamp"],
})
movies = movies.map(lambda x: x["movie_title"])

# housekeeping to prepare feature vocab
timestamps = np.concatenate(list(ratings.map(lambda x: x["timestamp"]).batch(100)))

max_timestamp = timestamps.max()
min_timestamp = timestamps.min()

timestamp_buckets = np.linspace(
    min_timestamp, max_timestamp, num=1000,
)

unique_movie_titles = np.unique(np.concatenate(list(movies.batch(1000))))
unique_user_ids = np.unique(np.concatenate(list(ratings.batch(1_000).map(
    lambda x: x["user_id"]))))

# Use the query model from the advanced preprocessing step earlier
class UserModel(keras.Model):

  def __init__(self, use_timestamps):
    super().__init__()

    self._use_timestamps = use_timestamps

    self.user_embedding = keras.Sequential([
        keras.layers.StringLookup(
            vocabulary=unique_user_ids, mask_token=None),
        keras.layers.Embedding(len(unique_user_ids) + 1, 32),
    ])

    if use_timestamps:
      self.timestamp_embedding = keras.Sequential([
          keras.layers.Discretization(timestamp_buckets.tolist()),
          keras.layers.Embedding(len(timestamp_buckets) + 1, 32),
      ])
      self.normalized_timestamp = keras.layers.Normalization(
          axis=None
      )

      self.normalized_timestamp.adapt(timestamps)

  def call(self, inputs):
    if not self._use_timestamps:
      return self.user_embedding(inputs["user_id"])

    return tf.concat([
        self.user_embedding(inputs["user_id"]),
        self.timestamp_embedding(inputs["timestamp"]),
        tf.reshape(self.normalized_timestamp(inputs["timestamp"]), (-1, 1)),
    ], axis=1)
  
# Use of timestamp features interacts with choice of traintest in a bad way
  # we split our data randomly instead of chronologically which is unrealistic

# adding time features lets it learn future patterns 
  
# Keeping the candidate model fixed again
  
class MovieModel(keras.Model):

  def __init__(self):
    super().__init__()

    max_tokens = 10_000

    self.title_embedding = keras.Sequential([
      keras.layers.StringLookup(
          vocabulary=unique_movie_titles, mask_token=None),
      keras.layers.Embedding(len(unique_movie_titles) + 1, 32)
    ])

    self.title_vectorizer = keras.layers.TextVectorization(
        max_tokens=max_tokens)

    self.title_text_embedding = keras.Sequential([
      self.title_vectorizer,
      keras.layers.Embedding(max_tokens, 32, mask_zero=True),
      keras.layers.GlobalAveragePooling1D(),
    ])

    self.title_vectorizer.adapt(movies)

  def call(self, titles):
    return tf.concat([
        self.title_embedding(titles),
        self.title_text_embedding(titles),
    ], axis=1)
  

# we can then combine both the UserModel and the MovieModel together
# this is a retrieval model
  
# note we also need to make sure query and candidate models output embeddings of compatible size
  # since we are varying sizes with more features the easiest way is to use a dense projection layer after each model
class MovielensModel(tfrs.models.Model):

  def __init__(self, use_timestamps):
    super().__init__()
    self.query_model = keras.Sequential([
      UserModel(use_timestamps),
      keras.layers.Dense(32)
    ])
    self.candidate_model = keras.Sequential([
      MovieModel(),
      keras.layers.Dense(32)
    ])
    self.task = tfrs.tasks.Retrieval(
        metrics=tfrs.metrics.FactorizedTopK(
            candidates=movies.batch(128).map(self.candidate_model),
        ),
    )

  def compute_loss(self, features, training=False):
    # We only pass the user id and timestamp features into the query model. This
    # is to ensure that the training inputs would have the same keys as the
    # query inputs. Otherwise the discrepancy in input structure would cause an
    # error when loading the query model after saving it.
    query_embeddings = self.query_model({
        "user_id": features["user_id"],
        "timestamp": features["timestamp"],
    })
    movie_embeddings = self.candidate_model(features["movie_title"])

    return self.task(query_embeddings, movie_embeddings)

  
# prepare data 
tf.random.set_seed(42)
shuffled = ratings.shuffle(100_000, seed=42, reshuffle_each_iteration=False)

train = shuffled.take(80_000)
test = shuffled.skip(80_000).take(20_000)

cached_train = train.shuffle(100_000).batch(2048)
cached_test = test.batch(4096).cache()

# baseline no timestamp features
model = MovielensModel(use_timestamps=False)
model.compile(optimizer=tf.keras.optimizers.Adagrad(0.1))

model.fit(cached_train, epochs=3)

train_accuracy = model.evaluate(
    cached_train, return_dict=True)["factorized_top_k/top_100_categorical_accuracy"]
test_accuracy = model.evaluate(
    cached_test, return_dict=True)["factorized_top_k/top_100_categorical_accuracy"]

print(f"Top-100 accuracy (train): {train_accuracy:.2f}.")
print(f"Top-100 accuracy (test): {test_accuracy:.2f}.")

# now with timestamp features
model = MovielensModel(use_timestamps=True)
model.compile(optimizer=tf.keras.optimizers.Adagrad(0.1))

model.fit(cached_train, epochs=3)

train_accuracy = model.evaluate(
    cached_train, return_dict=True)["factorized_top_k/top_100_categorical_accuracy"]
test_accuracy = model.evaluate(
    cached_test, return_dict=True)["factorized_top_k/top_100_categorical_accuracy"]

print(f"Top-100 accuracy (train): {train_accuracy:.2f}.")
print(f"Top-100 accuracy (test): {test_accuracy:.2f}.")