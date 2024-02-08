"""
An advantage to using a deep learning framework with recommender systems is the ability to build rich,
flexible feature representations. The first step in this process is preparing features since raw features
are not usable in the model.

For example:

User and item ids may be strings (titles, usernames) or large, noncontiguous integers (database IDs).
Item descriptions could be raw text.
Interaction timestamps could be raw Unix timestamps.
These need to be appropriately transformed in order to be useful in building models:

User and item ids have to be translated into embedding vectors: high-dimensional numerical representations that are adjusted during training to help the model predict its objective better.
Raw text needs to be tokenized (split into smaller parts such as individual words) and translated into embeddings.
Numerical features need to be normalized so that their values lie in a small interval around 0.
"""

"""
Tensorflow gives us the ability to make preprocessing as a part of the model rather than a separate step, (could be potential
to utilize this within the context of other models & frameworks as well)
"""

"""
guide for more preprocessing outside the context of recommender systems: https://www.tensorflow.org/guide/keras/preprocessing_layers 
"""

## Starting with initializing the movielens dataset
# at this point its become the norm

import pprint

import tensorflow_datasets as tfds

ratings = tfds.load("movielens/100k-ratings", split="train")

for x in ratings.take(1).as_numpy_iterator():
  pprint.pprint(x)

"""
Key features in our dataset: 

Movie title is useful as a movie identifier. (categorical)
User id is useful as a user identifier. (categorical)
Timestamps will allow us to model the effect of time. (continuous)
"""

## Categorical features to embeddings
# during model training embedding values get adjusted to help model make better predictions
# turning raw categorical data into embeddings is generally a two step process
# Firstly, we need to translate the raw values into a range of contiguous integers, normally 
#   by building a mapping (called a "vocabulary") that maps raw values ("Star Wars") to integers (say, 15).
# Secondly, we need to take these integers and turn them into embeddings.

## define vocab
import numpy as np
import tensorflow as tf
import keras.api._v2.keras as keras

movie_title_lookup = keras.layers.StringLookup()
# embedding layer does not have a vocab yet but can be built through adding our data    
movie_title_lookup.adapt(ratings.map(lambda x: x["movie_title"]))

print(f"Vocabulary: {movie_title_lookup.get_vocabulary()[:3]}")

# now we can translate raw tokens into embedding ID's
print(movie_title_lookup(["Star Wars (1977)", "One Flew Over the Cuckoo's Nest (1975)"]))

"""
returns: <tf.Tensor: shape=(2,), dtype=int64, numpy=array([ 1, 58])>
"""

"""
note on embedding layers containing OOV tokens:
An oov token is an 'out of vocabulary' token
These are used so the layer can handle categorical variables that are not present in the vocabulary 
practically speaking, this means the model can continue to learn and make recommendations even using features not seen that have
    not been seen during vocab initialization.
"""

"""
Via a string lookup layer we can configure multiple OOV indices making it so that any value not in the vocab can be deterministically 
    hashed to one of the OOV indices. The more OOV indices the less likely it is that two values get hashed to the same index.
Also if theres enough indices the model will be able to train about as well as a model with an explicit vocabulary without 
    the disdvantage of having to maintain the token list.

This can be taken to the extreme where we rely entirely on hashing with no vocab which can be implemented via a keras.layers.hashing() layer
ex)
# We set up a large number of bins to reduce the chance of hash collisions.
num_hashing_bins = 200_000

movie_title_hashing = keras.layers.Hashing(
    num_bins=num_hashing_bins
)
print(movie_title_hashing(["Star Wars (1977)", "One Flew Over the Cuckoo's Nest (1975)"]))

"""

# Now defining embeddings with our integer IDs
# embedding layers have 2 dimensions, the first being how many distinct categories we can embed and the second being how large the vector
# representing each can be 

"""
In the context of movies, 
when creating the embedding layer for movie titles, we are going to set the first value to the size of our title vocabulary
(or the number of hashing bins). The second is up to us: the larger it is, the higher the capacity of the model, but the slower
it is to fit and serve.
"""

movie_title_embedding = keras.layers.Embedding(
    # Let's use the explicit vocabulary lookup.
    input_dim=movie_title_lookup.vocab_size(),
    output_dim=32
)

# these two can be fastened together into a sequential layer which takes in raw text and outputs embeddings
movie_title_model = keras.Sequential([movie_title_lookup, movie_title_embedding])

# the same can also be applied to userEmbeddings
user_id_lookup = keras.layers.StringLookup()
user_id_lookup.adapt(ratings.map(lambda x: x["user_id"]))

user_id_embedding = keras.layers.Embedding(user_id_lookup.vocab_size(), 32)

user_id_model = keras.Sequential([user_id_lookup, user_id_embedding])

## Normalization of continuous features

# to normalize continuous features different methods can be used
# in the context of the movielens dataset, timestamp is too large to be used directly in the deep model
# Method #1: Standardization
"""
Standardization
Standardization rescales features to normalize their range by subtracting the feature's mean and dividing
by its standard deviation. It is a common preprocessing transformation. (think back to vintner Z-score application)

Easily accomplished by keras.layers.normalization

"""
timestamp_normalization = keras.layers.Normalization(
    axis=None
)
timestamp_normalization.adapt(ratings.map(lambda x: x["timestamp"]).batch(1024))

for x in ratings.take(3).as_numpy_iterator():
  print(f"Normalized timestamp: {timestamp_normalization(x['timestamp'])}.")

# Method #2: Discretization
"""
Another common transformation is to turn a continuous feature into a number of categorical features.
This makes good sense if we have reasons to suspect that a feature's effect is non-continuous.

To do this, we first need to establish the boundaries of the buckets we will use for discretization.
The easiest way is to identify the minimum and maximum value of the feature, and divide the resulting interval eq
"""
max_timestamp = ratings.map(lambda x: x["timestamp"]).reduce(
    tf.cast(0, tf.int64), tf.maximum).numpy().max()
min_timestamp = ratings.map(lambda x: x["timestamp"]).reduce(
    np.int64(1e9), tf.minimum).numpy().min()

timestamp_buckets = np.linspace(
    min_timestamp, max_timestamp, num=1000)

print(f"Buckets: {timestamp_buckets[:3]}")

# Given our buckets we can transform timestamps into embeddings 
timestamp_embedding_model = keras.Sequential([
  keras.layers.Discretization(timestamp_buckets.tolist()),
  keras.layers.Embedding(len(timestamp_buckets) + 1, 32)
])

for timestamp in ratings.take(1).map(lambda x: x["timestamp"]).batch(1).as_numpy_iterator():
  print(f"Timestamp embedding: {timestamp_embedding_model(timestamp)}.")

### Processing Text Features

"""
While the MovieLens dataset does not give us rich textual features, we can still use movie titles. 
This may help us capture the fact that movies with very similar titles are likely to belong to the same series.

The first transformation we need to apply to text is tokenization (splitting into constituent words or word-pieces),
followed by vocabulary learning, followed by an embedding.

The Keras tf.keras.layers.TextVectorization layer can do the first two steps for us:
"""

title_text = keras.layers.TextVectorization()
title_text.adapt(ratings.map(lambda x: x["movie_title"]))


for row in ratings.batch(1).map(lambda x: x["movie_title"]).take(1):
  print(title_text(row))

## Each title gets made into a sequence of tokens 

# can check that the learned vocab is correct with 
print(title_text.get_vocabulary()[40:45])


## Sample User model that puts it all together
class UserModel(keras.Model):

  def __init__(self):
    super().__init__()

    self.user_embedding = keras.Sequential([
        user_id_lookup,
        keras.layers.Embedding(user_id_lookup.vocab_size(), 32),
    ])
    self.timestamp_embedding = keras.Sequential([
      keras.layers.Discretization(timestamp_buckets.tolist()),
      keras.layers.Embedding(len(timestamp_buckets) + 2, 32)
    ])
    self.normalized_timestamp = keras.layers.Normalization(
        axis=None
    )

  def call(self, inputs):

    # Take the input dictionary, pass it through each input layer,
    # and concatenate the result.
    return tf.concat([
        self.user_embedding(inputs["user_id"]),
        self.timestamp_embedding(inputs["timestamp"]),
        tf.reshape(self.normalized_timestamp(inputs["timestamp"]), (-1, 1))
    ], axis=1)
  

## testing the user model
user_model = UserModel()

user_model.normalized_timestamp.adapt(
    ratings.map(lambda x: x["timestamp"]).batch(128))

for row in ratings.batch(1).take(1):
  print(f"Computed representations: {user_model(row)[0, :3]}")


## same for movie model
  
class MovieModel(keras.Model):

  def __init__(self):
    super().__init__()

    max_tokens = 10_000

    self.title_embedding = keras.Sequential([
      movie_title_lookup,
      keras.layers.Embedding(movie_title_lookup.vocab_size(), 32)
    ])
    self.title_text_embedding = keras.Sequential([
      keras.layers.TextVectorization(max_tokens=max_tokens),
      keras.layers.Embedding(max_tokens, 32, mask_zero=True),
      # We average the embedding of individual words to get one embedding vector
      # per title.
      keras.layers.GlobalAveragePooling1D(),
    ])

  def call(self, inputs):
    return tf.concat([
        self.title_embedding(inputs["movie_title"]),
        self.title_text_embedding(inputs["movie_title"]),
    ], axis=1)

movie_model = MovieModel()

movie_model.title_text_embedding.layers[0].adapt(
    ratings.map(lambda x: x["movie_title"]))

for row in ratings.batch(1).take(1):
  print(f"Computed representations: {movie_model(row)[0, :3]}")  