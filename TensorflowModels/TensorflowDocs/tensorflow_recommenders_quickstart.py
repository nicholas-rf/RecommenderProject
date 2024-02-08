from typing import Dict, Text

import numpy as np
import tensorflow as tf
import keras.api._v2.keras as keras
from tensorflow.python.framework.tensor import Tensor
import tensorflow_datasets as tfds
import tensorflow_recommenders as tfrs

"""
Side-Note: Regarding potential import issues with tensor-flow, there is a current bug which causes certain imports to not be resolved
A fix for this includes importing keras as 'import keras.api._v2.keras as keras'.
More info can be found at 'https://github.com/tensorflow/tensorflow/issues/53144#issuecomment-985179600'
"""

"""
This is the implementation of a retrieval model, which is more focused on speed and gathering a potential subset of items that
could be relevant to a user. Ranking models instead order the items that are gathered from retrieval for this 2 stage process.
"""

def create_vocabularies(movies, ratings):
    """
    Creates a vocabulary of integer indices for embedding layers in order to convert user ids

    Args:
        movies (MapDataset) : Contains a MapDataset element spec with movie titles
        ratings (MapDataset) : Contains a MapDataset element spec with movie_title and user_id as elements
    
    Returns:
        user_ids_vocabulary (tf.Keras.Layers) : A string lookup layer for the ratings
        movie_titles_vocabulary (tf.Keras.Layers) : A string lookup layer for the movies
    """

    # Create string lookup layers for both the movies and ratings
    user_ids_vocabulary = keras.layers.StringLookup(mask_token=None)
    user_ids_vocabulary.adapt(ratings.map(lambda x : x['user_id']))

    movie_titles_vocabulary = keras.layers.StringLookup(mask_token=None)
    movie_titles_vocabulary.adapt(movies)

    return user_ids_vocabulary, movie_titles_vocabulary, ratings, movies

"""
Notes on create_vocabularies

keras.layers.StringLookup functions similarly to textVectorization. However, its behavior is more simple as it just maps
strings to integer indices, which in my opinion is just creating simple embeddings for a lookup system. These layers must be
adapted to the specific vocabulary in order to work properly, as seen after its initialization.
The mask = None parameter in the StringLookup function I'm assuming is only there since it could be used for when a Mask is present
"""

def read_data():
    """
    Reads in the movielens dataset for 100k-ratings and 100k-movies for model training 
    """
    # Load in the ratings, and features for all available movies
    ratings = tfds.load('movielens/100k-ratings', split='train')
    movies = tfds.load('movielens/100k-movies', split='train')


    
    # Select the basic features from the movies
    ratings = ratings.map(lambda x : {
        'movie_title' : x['movie_title'],
        'user_id' : x['user_id']
    })
    movies = movies.map(lambda x : x['movie_title'])

    return create_vocabularies(movies, ratings)



"""
Notes on read_data() implementaton

Dataset is a Prefetch Dataset object which has element specs accessible as dictionary keys, as evidenced by the map function
which is utilizing a lambda function to extract specific features. Additionally, the actual elements themselves are made of
TensorSpec objects.
<_PrefetchDataset element_spec={'bucketized_user_age': TensorSpec(shape=(), dtype=tf.float32, name=None),
                  'movie_genres': TensorSpec(shape=(None,), dtype=tf.int64, name=None), 'movie_id': TensorSpec(shape=(),
                  dtype=tf.string, name=None), 'movie_title': TensorSpec(shape=(), dtype=tf.string, name=None),
                  'raw_user_age': ...

After accessing dataset with keys like ['movie_names'], we are left with a MapDataset element containing the keys spec

"""


# MODEL DEFINITION #

class MovieLensModel(tfrs.Model):
    # We derive from a custom base class to help reduce boilerplate under the hood, these are still plain Keras Models

    def __init__(
        self,
        user_model: keras.Model,
        movie_model: keras.Model,
        task: tfrs.tasks.Retrieval):
        super().__init__()

        # Set up the user and movie representations
        self.user_model = user_model
        self.movie_model = movie_model

        # Set up a retrieval task
        self.task = task

    def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> Tensor:
        # Define how loss is computed
        user_embeddings = self.user_model(features['user_id'])
        movie_embeddings = self.movie_model(features['movie_title'])

        return self.task(user_embeddings, movie_embeddings)
    
"""
Lot to write about here in terms of explainable stuff.
Starting with the first line 'class MovieLensModel(tfrs.Model)'; we start by intialializing a class supersetting the tensorflow recommenders
models. This is done since many recommender models are more complex so they don't fit into established supervised vs unsupervised paradigms.
With the tfrs.model we can have an easier time defining custom training and loss for models.

Moving onto the __init__. We create a model for the users and a model for the movies. According to CHATGPT, creating seperate models for users
and movies allows for each model to keep track of features which the recommendation system can then use to more efficiently handle the task.
In practice allegedly these are often combined into the two-tower architecture. Specifically for this case with matrix factorization however,
each model represents the user and items for the matrices! (I think). The task variable establishes what we are doing.

Then lastly the compute_loss function. Compute loss takes in features in a dictionary (I.E) the movie data with its key and a tensor, and then 
via the task calculates the loss given the embeddings. 

(Can write more about task later on)
"""

def define_models(user_ids_vocabulary, movie_titles_vocabulary, movies):
    """
    Defines a user and movie model for usage in the retrieval system of the recommender.
    """
    user_model = keras.Sequential([
        user_ids_vocabulary,
        keras.layers.Embedding(user_ids_vocabulary.vocab_size(), 64)
    ])

    
    movie_model = keras.Sequential([
        movie_titles_vocabulary,
        keras.layers.Embedding(movie_titles_vocabulary.vocab_size(), 64)
    ])

    # Define objectives    
    task = tfrs.tasks.Retrieval(metrics=tfrs.metrics.FactorizedTopK(
        movies.batch(128).map(movie_model)
    ))

    return user_model, movie_model, task

"""
Now to explain define_models. We define two sequential models, with sequential models it groups a stack of layers into a model. 
The sequential layer is made up of a list with the vocabulary and an embedding layer with the input and output dimension of 
either the user ids or movie titles. The task is a tfrs task for retrieval, which defaults to a categorical cross entropy loss function,
and uses a metrics of the movies batched by 128 and mapped to the movie model. These movies form a corpus of candidates.
More specifically batch moves movies into batches of 128, with map the movie model gets applied to each element
"""


def main():
    """

    """
    # we first create the data, which returns movies, ratings and vocabulary embeddings layers for both

    user_ids_vocabulary, movie_titles_vocabulary, ratings, movies = read_data()

    user_model, movie_model, task = define_models(user_ids_vocabulary, movie_titles_vocabulary, movies)

    # Create a retrieval model
    model = MovieLensModel(user_model, movie_model, task)
    model.compile(optimizer=keras.optimizers.Adagrad(0.5))

    # Train for 3 epochs
    model.fit(ratings.batch(4096), epochs=3)

    # Use brute-force search to set up retrieval using the trained representations
    index = tfrs.layers.factorized_top_k.BruteForce(model.user_model)
    index.index_from_dataset(
        movies.batch(100).map(lambda title : (title, model.movie_model(title)))
    )

    # Get some recommendations
    _, titles = index(np.array(["42"]))
    print(f'Top 3 recommendations for user 42: {titles[0, :3]}')


"""
Now to explain the final bit, the main function.

We start by initializing the vocabularies, ratings and movies. Then we create a user model, movie model and a task. Once complete
we create a full MovieLensModel with the user and movie models and the task, compile the model with an optimizer, fit the data 
and lastly generate some recommendations for a user based off of the result.

Diving into functions more specifically, compile configures the model for training with an optimizer, loss, metrics, etc
in this case we are using the Adagrad optimizer function.
Fitting the model just trains it, where in this case we are training it for 3 epochs using a batch size of 4096

Factorized top k brute force is an algorithm for getting the top k for each user from the user model. (I think, a little unsure here)

index.index_from_dataset() pulls the recommendations for the set of candidates specified by the movies.batch() call

Then creating a np array from the index can be used to get the titles of movies that are recommended to people. 
"""

if __name__ == "__main__":
    main()