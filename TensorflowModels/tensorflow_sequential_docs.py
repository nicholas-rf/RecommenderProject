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

