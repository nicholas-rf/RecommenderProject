# Necessary import
import os
import pprint
import tempfile

from typing import Dict, Text

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import keras
import tensorflow_recommenders as tfrs
from tensorflow.python.framework.tensor import Tensor

# Ratings data.
ratings = tfds.load("movielens/100k-ratings", split="train")

# Features of all the available movies.
movies = tfds.load("movielens/100k-movies", split="train")

# Print ratings
for x in ratings.take(1).as_numpy_iterator():
  pprint.pprint(x)

# Print out a movie
for x in movies.take(1).as_numpy_iterator():
  pprint.pprint(x)

# Part of tutorial that strictly focuses on user ids and movie titles
ratings = ratings.map(lambda x: {
  "movie_title": x["movie_title"],
  "user_id": x["user_id"],
})
movies = movies.map(lambda x: x["movie_title"])

# In industrial recommender systems training and testing splits are usually predicated on time intervals
# I.E data before time T is used to train, and data after time T is used to test

# Model splitting
tf.random.set_seed(42)
shuffled = ratings.shuffle(100_000, seed=42, reshuffle_each_iteration=False)

# Figure out unique user ids and move titles that are present for embeddings
train = shuffled.take(80_000)
test = shuffled.skip(80_000).take(20_000)

movie_titles = movies.batch(1_000)
user_ids = ratings.batch(1_000_000).map(lambda x: x["user_id"])

unique_movie_titles = np.unique(np.concatenate(list(movie_titles)))
unique_user_ids = np.unique(np.concatenate(list(user_ids)))

unique_movie_titles[:10]

## Moving on to model implementation

# First step is choosing embedding dimension
embedding_dimension = 32

# Higher values mean more accurate models however longer fitting times and more prone to overfitting

# Set up the user model as a keras sequential model with a string lookup layer which converts ids to numeric values
# And then an embedding layer which creates embeddings from the vocabulary
user_model = keras.Sequential([
  keras.layers.StringLookup(
      vocabulary=unique_user_ids, mask_token=None),
  # We add an additional embedding to account for unknown tokens.
  keras.layers.Embedding(len(unique_user_ids) + 1, embedding_dimension)
])

# Simple model like this corresponds to matrix factorization approach, can add complexity by adding further models
# however for this current use case that is arbitrary, also if adding more models for complexity,
# as long as it ends in an embedding layer thats good

# Same for candidate tower
movie_model = keras.Sequential([
  keras.layers.StringLookup(
      vocabulary=unique_movie_titles, mask_token=None),
  keras.layers.Embedding(len(unique_movie_titles) + 1, embedding_dimension)
])

# Now we decide on metrics for recommendations,
# metrics.FactorizedTopK requires only one argument, the dataset which has been through embeddings

metrics = tfrs.metrics.FactorizedTopK(
  candidates=movies.batch(128).map(movie_model)
)

# Now we focus on loss metrics for the recommender, TFRS has several loss layers but for this model 
# Well use the retrieval task which bundles together loss function and metric computation
task = tfrs.tasks.Retrieval(
  metrics=metrics
)

# The task itself takes in the query and candidate embeddings as arguments and returns computed loss
# Used in the models training loop

## The full model  

# building the model is easier since we can utilize the tensorflow recommenders base model class tfrs.models.Model
# Meaning all we need to add is the __init__ and the compute loss function taking in raw features 
# and outputting a loss value
# The base model takes care of creating the training loop

# Nick comment
# As visible below The init takes in our user and movie towers, and sets up self models as the towers
# Also we can see that the compute loss function as appended with the user and movie models embeddings in midn
class MovielensModel(tfrs.Model):

  def __init__(self, user_model, movie_model):
    super().__init__()
    self.movie_model: keras.Model = movie_model
    self.user_model: keras.Model = user_model
    self.task: keras.layers.Layer = task

  def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
    # We pick out the user features and pass them into the user model.
    user_embeddings = self.user_model(features["user_id"])
    # And pick out the movie features and pass them into the movie model,
    # getting embeddings back.
    positive_movie_embeddings = self.movie_model(features["movie_title"])

    # The task computes the loss and the metrics.
    return self.task(user_embeddings, positive_movie_embeddings)
  

# below is an example of not utilizing the base class from tensorflow recommenders, and instead making one via
# the keras API ( might be helpful for more creativity later on (icing) )
  
class NoBaseClassMovielensModel(tf.keras.Model):

  def __init__(self, user_model, movie_model):
    super().__init__()
    self.movie_model: keras.Model = movie_model
    self.user_model: keras.Model = user_model
    self.task: keras.layers.Layer = task

  def train_step(self, features: Dict[Text, tf.Tensor]) -> tf.Tensor:

    # Set up a gradient tape to record gradients.
    with tf.GradientTape() as tape:

      # Loss computation.
      user_embeddings = self.user_model(features["user_id"])
      positive_movie_embeddings = self.movie_model(features["movie_title"])
      loss = self.task(user_embeddings, positive_movie_embeddings)

      # Handle regularization losses as well.
      regularization_loss = sum(self.losses)

      total_loss = loss + regularization_loss

    gradients = tape.gradient(total_loss, self.trainable_variables)
    self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

    metrics = {metric.name: metric.result() for metric in self.metrics}
    metrics["loss"] = loss
    metrics["regularization_loss"] = regularization_loss
    metrics["total_loss"] = total_loss

    return metrics

  def test_step(self, features: Dict[Text, tf.Tensor]) -> tf.Tensor:

    # Loss computation.
    user_embeddings = self.user_model(features["user_id"])
    positive_movie_embeddings = self.movie_model(features["movie_title"])
    loss = self.task(user_embeddings, positive_movie_embeddings)

    # Handle regularization losses as well.
    regularization_loss = sum(self.losses)

    total_loss = loss + regularization_loss

    metrics = {metric.name: metric.result() for metric in self.metrics}
    metrics["loss"] = loss
    metrics["regularization_loss"] = regularization_loss
    metrics["total_loss"] = total_loss

    return metrics  
  
# The focus in the tutorials on the TFRS website is on setting up models and not the boilerplate as visible above


## Fitting and evaluating 
model = MovielensModel(user_model, movie_model)
model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))

# shuffling batching and chache the training data and evaluation data
cached_train = train.shuffle(100_000).batch(8192).cache()
cached_test = test.batch(4096).cache()

# Train the model
model.fit(cached_train, epochs=3)

# Could try and figure out how to utilize tensorboard to track model performance 

# Evaluating the performance of the model
model.evaluate(cached_test, return_dict=True)


## Making predictions

# Create a model that takes in raw query features, and
index = tfrs.layers.factorized_top_k.BruteForce(model.user_model)
# recommends movies out of the entire movies dataset.
index.index_from_dataset(
  tf.data.Dataset.zip((movies.batch(100), movies.batch(100).map(model.movie_model)))
)

# Get recommendations.
_, titles = index(tf.constant(["42"]))
print(f"Recommendations for user 42: {titles[0, :3]}")

# THE REST OF THE TUTORIAL GOES INTO SAVING MODELS AND APPLYING THEM IRL # 