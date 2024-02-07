# Sequential models utilize previous users interactions to predict the next interaction
# Order of items in sequence matters so a recurrent neural network is utilized to model the 
# sequential relationship

import os
import pprint
import tempfile

from typing import Dict, Text

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_recommenders as tfrs
import keras.api._v2.keras as keras

## Preparing the dataset with tensorflow data generating ability with TensorFlow Lite On-device Recommendation reference app.
# wget -nc https://raw.githubusercontent.com/tensorflow/examples/master/lite/examples/recommendation/ml/data/example_generation_movielens.py
# python -m example_generation_movielens  --data_dir=data/raw  --output_dir=data/examples  --min_timeline_length=3  --max_context_length=10  --max_context_movie_genre_length=10  --min_rating=2  --train_data_fraction=0.9  --build_vocabs=False
## 

## The next lines load in the dataset and set it up as a TFRecordDataset
## The dataset contains a sequence of context movie IDs with a labelled movie ID plus some context features like
## Movie year, rating and genre

## This doc only contains information for leveraging context features 

train_filename = "./data/examples/train_movielens_1m.tfrecord"
train = tf.data.TFRecordDataset(train_filename)

test_filename = "./data/examples/test_movielens_1m.tfrecord"
test = tf.data.TFRecordDataset(test_filename)

feature_description = {
    'context_movie_id': tf.io.FixedLenFeature([10], tf.int64, default_value=np.repeat(0, 10)),
    'context_movie_rating': tf.io.FixedLenFeature([10], tf.float32, default_value=np.repeat(0, 10)),
    'context_movie_year': tf.io.FixedLenFeature([10], tf.int64, default_value=np.repeat(1980, 10)),
    'context_movie_genre': tf.io.FixedLenFeature([10], tf.string, default_value=np.repeat("Drama", 10)),
    'label_movie_id': tf.io.FixedLenFeature([1], tf.int64, default_value=0),
}

def _parse_function(example_proto):
  return tf.io.parse_single_example(example_proto, feature_description)

train_ds = train.map(_parse_function).map(lambda x: {
    "context_movie_id": tf.strings.as_string(x["context_movie_id"]),
    "label_movie_id": tf.strings.as_string(x["label_movie_id"])
})

test_ds = test.map(_parse_function).map(lambda x: {
    "context_movie_id": tf.strings.as_string(x["context_movie_id"]),
    "label_movie_id": tf.strings.as_string(x["label_movie_id"])
})

for x in train_ds.take(1).as_numpy_iterator():
  pprint.pprint(x)


### Now train and test has been set up with a shape of 10 as the length of context features for the example generation step
# Before moving to the model we also get a vocabulary for our movie ids
movies = tfds.load("movielens/1m-movies", split='train')
movies = movies.map(lambda x: x["movie_id"])
movie_ids = movies.batch(1_000)
unique_movie_ids = np.unique(np.concatenate(list(movie_ids)))

## Two tower architecture is still utilized in this case 
# the query tower (the user tower) specifically used a Gated Recurrent Unit (GRU) layer to encode the sequence of movies

embedding_dimension = 32

query_model = keras.Sequential([
    keras.layers.StringLookup(
      vocabulary=unique_movie_ids, mask_token=None),
    keras.layers.Embedding(len(unique_movie_ids) + 1, embedding_dimension), 
    keras.layers.GRU(embedding_dimension),
])

candidate_model = keras.Sequential([
  keras.layers.StringLookup(
      vocabulary=unique_movie_ids, mask_token=None),
  keras.layers.Embedding(len(unique_movie_ids) + 1, embedding_dimension)
])

## Metrics and task are defined similarly to the retrieval model 

metrics = tfrs.metrics.FactorizedTopK(
  candidates=movies.batch(128).map(candidate_model)
)

task = tfrs.tasks.Retrieval(
  metrics=metrics
)

class Model(tfrs.Model):

    def __init__(self, query_model, candidate_model):
        super().__init__()
        self._query_model = query_model
        self._candidate_model = candidate_model

        self._task = task

    def compute_loss(self, features, training=False):
        watch_history = features["context_movie_id"]
        watch_next_label = features["label_movie_id"]

        query_embedding = self._query_model(watch_history)       
        candidate_embedding = self._candidate_model(watch_next_label)

        return self._task(query_embedding, candidate_embedding, compute_metrics=not training)
    
## Now we just do standard fitting as usual 
model = Model(query_model, candidate_model)
model.compile(optimizer=keras.optimizers.Adagrad(learning_rate=0.1))

cached_train = train_ds.shuffle(10_000).batch(12800).cache()
cached_test = test_ds.batch(2560).cache()

model.fit(cached_train, epochs=3)

## Eval step
model.evaluate(cached_test, return_dict=True)


