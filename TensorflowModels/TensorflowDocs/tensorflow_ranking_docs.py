

## Blurb from tensorflows website on ranking vs retreiving 
# The ranking stage takes the outputs of the retrieval model and fine-tunes them to select the best
# possible handful of recommendations. Its task is to narrow down the set of items the user may be 
# interested in to a shortlist of likely candidates.

## emphasis of this doc 
# Get our data and split it into a training and test set.
# Implement a ranking model.
# Fit and evaluate it.

import os
import pprint
import tempfile

from typing import Dict, Text

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_recommenders as tfrs
import keras


## Dataset prep

ratings = tfds.load("movielens/100k-ratings", split="train")

ratings = ratings.map(lambda x: {
    "movie_title": x["movie_title"],
    "user_id": x["user_id"],
    "user_rating": x["user_rating"]
})

# splitting 

tf.random.set_seed(42)
shuffled = ratings.shuffle(100_000, seed=42, reshuffle_each_iteration=False)

train = shuffled.take(80_000)
test = shuffled.skip(80_000).take(20_000)

# Get unique ids and movies 
movie_titles = ratings.batch(1_000_000).map(lambda x: x["movie_title"])
user_ids = ratings.batch(1_000_000).map(lambda x: x["user_id"])

unique_movie_titles = np.unique(np.concatenate(list(movie_titles)))
unique_user_ids = np.unique(np.concatenate(list(user_ids)))

## implementing a model 

# Ranking models do not face the same efficiency issues that retrieval models do, therefore the freedom of choice
# amongst architectures is larger 
# A model of many stacked dense layers is a common architecture for ranking tasks

class RankingModel(keras.Model):

  def __init__(self):
    # Similar syntax to tensorflow_retrieval_docs.py ## 
    super().__init__()
    embedding_dimension = 32

    # Compute embeddings for users.
    self.user_embeddings = keras.Sequential([
      keras.layers.StringLookup(
        vocabulary=unique_user_ids, mask_token=None),
      keras.layers.Embedding(len(unique_user_ids) + 1, embedding_dimension)
    ])

    # Compute embeddings for movies.
    self.movie_embeddings = keras.Sequential([
      keras.layers.StringLookup(
        vocabulary=unique_movie_titles, mask_token=None),
      keras.layers.Embedding(len(unique_movie_titles) + 1, embedding_dimension)
    ])
    ## within the init step user and movie tower embeddings are created ## 

    # Compute predictions.
    self.ratings = keras.Sequential([
      # Learn multiple dense layers.
      keras.layers.Dense(256, activation="relu"),
      keras.layers.Dense(64, activation="relu"),
      # Make rating predictions in the final layer.
      ## Why is the output of size 1? ##
      keras.layers.Dense(1)
  ])

  def call(self, inputs):

    user_id, movie_title = inputs

    user_embedding = self.user_embeddings(user_id)
    movie_embedding = self.movie_embeddings(movie_title)

    return self.ratings(tf.concat([user_embedding, movie_embedding], axis=1))
  
## This model takes in user ids and movie titles and outputs a predicted rating ## 
  
# visible in example below # 
RankingModel()((["42"], ["One Flew Over the Cuckoo's Nest (1975)"]))

# Set up loss metric task 
task = tfrs.tasks.Ranking(
  loss = keras.losses.MeanSquaredError(),
  metrics=[keras.metrics.RootMeanSquaredError()]
)

# Task takes in true and predicted values and outputs the computed loss, utilized in the models training loop

## Putting it all into the model 
class MovielensModel(tfrs.models.Model):

  def __init__(self):
    super().__init__()
    self.ranking_model: keras.Model = RankingModel()
    self.task: keras.layers.Layer = tfrs.tasks.Ranking(
      loss = keras.losses.MeanSquaredError(),
      metrics = [keras.metrics.RootMeanSquaredError()]
    )

  def call(self, features: Dict[str, tf.Tensor]) -> tf.Tensor:
    return self.ranking_model(
        (features["user_id"], features["movie_title"]))

  def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
    labels = features.pop("user_rating")

    rating_predictions = self(features)

    # The task computes the loss and the metrics.
    return self.task(labels=labels, predictions=rating_predictions)
  
# Fitting and evaluating ## 
model = MovielensModel()    
model.compile(optimizer=keras.optimizers.Adagrad(learning_rate=0.1))

# Cache and shuffle the train and testing datasets
cached_train = train.shuffle(100_000).batch(8192).cache()
cached_test = test.batch(4096).cache()

# Train the model 
model.fit(cached_train, epochs=3)

# evaluate the model 
model.evaluate(cached_test, return_dict=True)

## Testing the ranking model 
test_ratings = {}
test_movie_titles = ["M*A*S*H (1970)", "Dances with Wolves (1990)", "Speed (1994)"]
for movie_title in test_movie_titles:
  test_ratings[movie_title] = model({
      "user_id": np.array(["42"]),
      "movie_title": np.array([movie_title])
  })

print("Ratings:")
for title, score in sorted(test_ratings.items(), key=lambda x: x[1], reverse=True):
  print(f"{title}: {score}")