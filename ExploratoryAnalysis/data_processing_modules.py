from datetime import datetime
from collections import Counter
from tqdm import tqdm
import pandas as pd
import numpy as np
import clustering_modules as cm

"""
This module contains several methods used for feature extraction and to process and transform data for use in exploratory data analysis. 
"""

def data_to_csv(behaviors : bool, fpath : str) -> None:
    """
    Takes a tab seperated variable file from the MIND dataset, adds columns to it, and exports it as a CSV.
    This is only ran once to set up the CSVs that will be used throughout the rest of the project.

    Args:
        behaviors (bool) : A boolean which signifies that the incomming tsv is either the behaviors tsv or the news tsv.
        fpath (str) : The path to the directory that the csv will be output to.

    Returns:
        None
    """
    
    # If we are changing the format of the behaviors tsv, then use that one.
    if behaviors:

        # Column names to be added.
        behaviors_columns = ['impression_id', 'user_id', 'time', 'history', 'impressions']

        # Read in the tsv adding names and then export it to a csv.
        df = pd.read_csv(fpath, sep='\t', names=behaviors_columns)
        df.to_csv('../MIND_large/csv/behaviors.csv')

    # If we are are not changing the format of the behaviors tsv, use the news csv.
    else:

        # Column names to be added.
        news_columns = ['news_id', 'category', 'sub_category', 'title', 'abstract', 'url', 'title_entities', 'abstract_entities']

        # Read in the tsv adding names and then export it to a csv.
        df = pd.read_csv(fpath, sep='\t', names=news_columns)
        df.to_csv('../MIND_large/csv/news.csv')

def clean_impression(impression : str = 'N55689-1') -> dict:
    """ 
    Cleans up a user impression for its characteristics. 

    Args:
        impression (str) : A users impression on a recommended article.
    
    Returns:
        impression_info (dict) : A dictionary containing keys for the rating and article in the impression.
    """

    # Split the impression by '-'.
    impression_info = impression.split('-')

    # Return a dictionary with the articleID and the impression score.
    return {'score':impression_info[1], 'article_ID':impression_info[0]}  

def create_popularity_dfs(news_frame : pd.DataFrame, behaviors_frame : pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """ 
    Creates dataframes to get the popularity of categories and articles through a users history and clickthrough rates.
    This is accomplished by iterating through the users in the behaviors dataframe, and incrementing dictionaries of articles and categories
    that get placed into dataframes.
    
    Args:
        news_frame (pd.DataFrame) : The news dataframe so that the index can be set to the news ID and used when querying user history.
        behaviors_frame (pd.DataFrame) : The behaviors dataframe so that users last interaction history can be used to understand category popularity.
    
    Returns:
        category_popularity (pd.DataFrame) : A dataframe with categories as columns and popularity as a row.
        article_popularity (pd.DataFrame) : A dataframe with articles as rows and popularity as a column.
    """

    # Create a copy of the news dataframe where the index is the news_id.
    copynews = news_frame.set_index('news_id')

    # Create empty dictionaries that get filled while iterating through the dataset.
    article_popularity_impression = {article: 0 for article in news_frame['news_id']}
    article_popularity_history = {article: 0 for article in news_frame['news_id']}
    category_popularity_impression = {category: 0 for category in pd.unique(copynews['category'])}
    category_popularity_history = {category: 0 for category in pd.unique(copynews['category'])}
    user_ids = []

    # Iterate through every row of the columns history and impressions in the dataframe. Tqdm is used here to show progress of iteration.
    for user_id, history, impressions in tqdm(zip(behaviors_frame['user_id'], behaviors_frame['history'], behaviors_frame['impressions']),
                        total=len(behaviors_frame['user_id']), desc="Iterating Over Behaviors"):
        # If our history is not a NaN.
        if type(history) != float:
            if user_id not in user_ids:
                # Split the history into news IDs
                for news_id in history.split():
                    # Locate the article and access its category, then increment the metrics.
                    category = copynews.loc[news_id]['category']
                    category_popularity_history[category] += 1
                    article_popularity_history[news_id] += 1
                user_ids.append(user_id)

        # If our impression is not a NaN.
        if type(impressions) != float:

            # Access all impressions.
            for impression in impressions.split():

                # Clean up the impression so that it's easier to parse.
                impression_info = clean_impression(impression)

                # If the impression signifies a clickthrough increment the metrics.
                if impression_info['score'] == '1':
                    article_popularity_impression[impression_info['article_ID']] += 1
                    category = copynews.loc[impression_info['article_ID']]['category']
                    category_popularity_impression[category] += 1
    
    # Turn the article popularities into lists for easier dataframe conversion.
    article_popularity_impression = list(article_popularity_impression.items())
    article_popularity_history = list(article_popularity_history.items())
    
    # Return a touple of 4 dataframes with the popularity information.
    return (pd.DataFrame(data=category_popularity_history, index=[1]),
            pd.DataFrame(data=category_popularity_impression, index=[1]),
            pd.DataFrame(data=article_popularity_history, columns=['article', 'popularity_history']),
            pd.DataFrame(data=article_popularity_impression, columns=['article', 'popularity_impression'])) 

def create_popularity_csvs(news : pd.DataFrame, behaviors : pd.DataFrame) -> None:
    """
    Extracts popularity for categories and articles from user history and impressions then exports them to csvs.
    This is done by using create_popularity_dfs which creates four separate dataframes which represent article
    popularity and category popularity. The article and category pairs are then concatenated and exported for usage within
    exploratory data analysis and news feature extraction.
    
    Args:
        news (pd.DataFrame) : The dataframe containing the data from the news csv.
        behaviors (pd.DataFrame) : The dataframe containing the data from the behaviors csv.
    
    Returns:
        None
    """

    # Create the dataframes with information about how popular each category is.
    print("Creating Popularity Counts")
    cat_pop_hist, cat_pop_imp, art_pop_hist, art_pop_imp = create_popularity_dfs(news, behaviors)

    # Merge the article popularities onto the news dataframe.
    art_pop_imp.rename(columns={'article':'news_id'}, inplace=True)
    art_pop_hist.rename(columns={'article':'news_id'}, inplace=True)
    art_pop = art_pop_imp.merge(art_pop_hist)
    news = news.merge(art_pop)

    # Delete old article popularity dataframes to save memory.
    del art_pop
    del art_pop_imp
    del art_pop_hist

    # Reset the index of the category popularity to create an index column.
    cat_test = cat_pop_imp.reset_index()
    cat_test2 = cat_pop_hist.reset_index()
    
    # Rename the index columns to popularity_type for understandability.
    cat_test['index'] = cat_test['index'].apply(lambda x : 'impressions')
    cat_test.rename(columns={'index' : 'popularity_type'}, inplace=True)
    cat_test2['index'] = cat_test2['index'].apply(lambda x : 'history')
    cat_test2.rename(columns={'index' : 'popularity_type'}, inplace=True)

    # Concatenate the 2 dataframes on rows.
    categories = pd.concat([cat_test, cat_test2], axis=0)
    
    # Output the CSVs to the respective folder.
    print("Outputting To Csv")
    news.to_csv("../MIND_large/csv/news_with_popularity.csv")
    categories.to_csv('../MIND_large/csv/category_with_popularity.csv')
        
def decompose_interactions(news : pd.DataFrame, behaviors : pd.DataFrame) -> pd.DataFrame:
    """
    Iterates through user interactions to create a tensorflow compatible dataset comprised of multiple rows per user, each row detailing
    their interaction, and relevant information about that interaction.
    
    Args:
        news (pd.DataFrame) : The dataframe containing the data from the news csv.
        behaviors (pd.DataFrame) : The dataframe containing the data from the behaviors csv.
    
    Returns:
        results (pd.DataFrame) : The dataframe containing users and their rankings. The columns of the dataframe are
        user_id, article_id, rating, rating_type, category, sub_category, timestamp. 
    """
    data = {'user_id' : [], 'time' : [], 'news_id' : [], 'category' : [], 'sub_category' : [], 'title' : [], 'abstract' : [], 'interaction_type' : [], 'score' : []}
    print(news.columns)
    print(type(news))
    # Setting the index of news to news_id so that lookup for information can be done.
    copynews = news.copy()
    copynews = copynews.set_index('news_id')
    
    # Creating an update_data function for convenience.
    def update_data(data : dict, user_id : str, time_stamp : str, news_id : str, interaction_type : str, score : int) -> dict:
        """
        Updates the data dictionary with a users interaciton data.
        """
        # Updating all relevant keys in the dictionary.
        data['user_id'].append(user_id)
        data['time'].append(time_stamp)
        data['news_id'].append(news_id)
        data['category'].append(copynews.loc[news_id]['category'])
        data['sub_category'].append(copynews.loc[news_id]['sub_category'])
        data['title'].append(copynews.loc[news_id]['title'])
        data['abstract'].append(copynews.loc[news_id]['abstract'])
        data['interaction_type'].append(interaction_type)
        data['score'].append(score)
        return data

    # Initialize a list to store user IDs as a way of checking if a users history has already been read into the dataset or not.
    seen = []

    # Iterating through all relevant information. Again we use tqdm to output a progress bar.
    print('Starting loop')
    for user_id, history, impressions, time_stamp in tqdm(zip(behaviors['user_id'], behaviors['history'], behaviors['impressions'], behaviors['hour']), total=len(behaviors['user_id'])):

        # Iterate over the users history if they have not been seen yet.
        if user_id not in seen:
            if type(history) != float:
                for news_id in history.split():
                    data = update_data(data, user_id, time_stamp, news_id, 'history', 1)
                
                # Add the user ID to the seen list.
                seen.append(user_id)
        
        # Iterate over the users impressions.
        for impression in impressions.split():

            # Clean the impression and then update the data accordingly.
            impression_result = clean_impression(impression)
            if impression_result['score'] == 1:
                data = update_data(data, user_id, time_stamp, impression_result['article_ID'], 'impression', 1)   

            else:
                data = update_data(data, user_id, time_stamp, impression_result['article_ID'], 'impression', 0)   

    # Returning a dataframe with the dictionary as the data.
    return pd.DataFrame(data=data)

def create_interaction_counts(behaviors):
    """
    Iterates over the behaviors data to extract the category counts for each interaction in order to analyze popularity of categories at different times of day.
    News, behaviors and categories with popularity are loaded in and then category popularity counts in both user history and impressions are extracted.

    Args:
        behaviors (pd.DataFrame) : The base behaviors dataset.
    
    Returns:
        None : Instead of returning a value this function stores the output as a csv. 
    """
    
    # Load in the datasets and set the index of the news dataset to news id    
    news = pd.read_csv('../MIND_large/csv/news.csv')
    copynews = news.set_index('news_id')
    category_popularity = pd.read_csv('../MIND_large/csv/category_with_popularity.csv')
    category_popularity.drop(columns=['Unnamed: 0'], inplace=True)

    def get_interaction_popularity(row, history=True):
        """
        Gets category popularity counts for a given interaction.

        Args:
            row (pd.DataFrame) : A row containing information.
            copynews (pd.DataFrame) : The news dataset with the index set to news ID. Used for lookup of article category.
            history (boolean) : A boolean that signifies whether or not the history column should be used.

        Returns:
            category_popularity (dict) : A dictionary containing popularity counts for a given user impression.
        """

        # Create dictionaries to store interacted categories for each user by history and their clickthrough rates.
        category_popularity_impression = {category: 0 for category in pd.unique(copynews['category'])}
        category_popularity_history = {category: 0 for category in pd.unique(copynews['category'])}

        # Get the history and impressions of the user in the row.
        if history:
            history=row['history']
            # If our history is not a NaN.
            if type(history) != float:

                # Split the history into news IDs
                for news_id in history.split():

                    # Locate the article and access its category, then increment the metrics.
                    category = copynews.loc[news_id]['category']
                    category_popularity_history[category] += 1
        else:
            impressions=row['impressions']
            # If the impression is not a NaN.
            if type(impressions) != float:
                # Access all impressions.
                for impression in impressions.split():

                    # Clean up the impression so that it's easier to parse.
                    impression_info = clean_impression(impression)

                    # If the impression signifies a clickthrough increment the metrics.
                    if impression_info['score'] == '1':
                        category = copynews.loc[impression_info['article_ID']]['category']
                        category_popularity_impression[category] += 1

        return category_popularity_history if history else category_popularity_impression

    # Get popularity counts for user histories, since user histories are global this is more an indicator of users with what history are interacting at what time.
    print('Starting history popularity')
    behaviors[category_popularity.columns.to_list()[1:]] = behaviors.apply(lambda row : get_interaction_popularity(row, True), axis='columns', result_type='expand')
    behaviors.rename(columns={column : column + "_history" for column in category_popularity.columns.to_list()}, inplace=True)

    # Get popularity counts for user impressions.
    print('Starting impression popularity')
    behaviors[category_popularity.columns.to_list()[1:]] = behaviors.apply(lambda row : get_interaction_popularity(row, False), axis='columns', result_type='expand')
    behaviors.rename(columns={column : column + "_impression" for column in category_popularity.columns.to_list()}, inplace=True)

    behaviors.to_csv("../MIND_large/csv/behaviors_with_individual_counts.csv")

def modify_hourly(behaviors):
    """
    Bins the time column of behaviors into hours so that insights on how time of day affects category popularity can be examined.
    
    Args:
        behaviors (pd.DataFrame) : Behaviors dataframe with impression specific category popularity counts.

    Returns:
        behaviors (pd.DataFrame) : Behaviors dataframe modified to include a binned time column. 
    """
    # Can be printed to determine the minimum and maximum timeframe for the data.
    # times = [behaviors['time'].max(), behaviors['time'].min()]

    # Make the time column into a datetime object.
    behaviors['time'] = pd.to_datetime(behaviors['time'])

    # going to want to set it so that time is only representing the date of the week and not hours, so cut_points can join the date with 00:00:00 for hourly predictions
    
    cut_points = pd.date_range(start='2019-11-15 00:00:00', end='2019-11-16 00:00:00', freq='h') # hourly ranges for the time of the behaviors dataset
    # going to want to adjust cutpoints so that we are specifically thinking of hours from 1 - 24 with 24 being midnight (0)
    
    # Create labels for the bins.
    bins_str = cut_points.astype(str).values
    labels = ['({}, {}]'.format(bins_str[i-1], bins_str[i]) for i in range(1, len(bins_str))]
    
    # Apply the bins to the time column.
    behaviors['hour'] = pd.cut(behaviors['time'], cut_points, labels=labels, include_lowest=True)
    
    # Modify the times so that it is only hours.
    behaviors['hour'] = behaviors['hour'].apply(lambda time_string : time_string.split(" ")[-1][:2])
    return behaviors

def create_hourly_long():
    """
    Creates long format dataframes with hourly popularity counts for data visualizations and as a potential feature for popularity. 
    This function utilizes the 'modify_hourly' function which bins the data into specific hours based off of the time frame present in
    the dataset.

    Args:
        None
    
    Returns:
        behaviors2, unique_user_histories2 (pd.DataFrame) : Two pandas dataframes containing long format popularity counts for later graphing.
    """
    # Load in the behaviors with individual counts dataset and the news dataset.    
    # Behaviors with individual counts is created with create_interaction_counts
    behaviors = pd.read_csv('../MIND_large/csv/behaviors_with_individual_counts.csv')
    news = pd.read_csv('../MIND_large/csv/news.csv')

    # Drop duplicate user ID entries and subset for the user id, time and all history popularity counts.
    unique_user_histories = behaviors.drop_duplicates(subset='user_id')[['user_id', 'time'] + [category + '_history' for category in news['category'].unique()]]
    
    # Drop the history popularity columns.
    behaviors = behaviors.drop(columns=[category + '_history' for category in news['category'].unique()])
    
    # Change datetime values into specific hours. 
    behaviors = modify_hourly(behaviors)
    unique_user_histories = modify_hourly(unique_user_histories)

    # Drop uneeded columns and group by hour applying a summation to all interaction counts per hour.
    behaviors = behaviors.drop(columns=['Unnamed: 0.1', 'Unnamed: 0', 'impression_id', 'history', 'impressions'])   
    behaviors2 = behaviors.drop(columns=['time', 'user_id']).groupby('hour', observed=False).agg('sum').reset_index()
    unique_user_histories2 = unique_user_histories.drop(columns=['time', 'user_id']).groupby('hour', observed=False).agg('sum').reset_index()

    # Create a list of column names for each frame.
    cols = ['lifestyle', 'health', 'news', 'sports', 'weather', 'entertainment', 'autos', 'travel', 'foodanddrink', 'tv', 'finance', 'movies', 'video', 'music', 'kids', 'middleeast']
    impression_ = []
    history_ = []
    for col in cols:
        impression_.append(col + '_impression')
        history_.append(col + '_history')

    # Apply normalization to the values.
    unique_user_histories2['history_div'] = unique_user_histories2[history_].apply(lambda x : sum(x), axis=1)
    behaviors2['impression_div'] = behaviors2[impression_].apply(lambda x : sum(x), axis=1)
    unique_user_histories2[history_] = unique_user_histories2.apply(lambda x : x[history_] / x['history_div'], axis=1)
    behaviors2[impression_] = behaviors2.apply(lambda x : x[impression_] / x['impression_div'], axis=1)
    return behaviors2, unique_user_histories2

def create_user_taste_profile(df):    
    """
    Uses the tensorflow compatible dataset to create the user feature dataframe. The user feature dataframe
    contains counts for each category and sub category so that user preference can be utilized later on.

    Args:
        df (pd.DataFrame) : The tensorflow compatible dataset as a dataframe for modification.

    Returns:
        users (pd.DataFrame) : The user feature dataframe.
    """
    # Subset the dataframe for only interactions with a positive score.
    subset = df[df['score'] == 1]

    # Group by user ID summing all scores, then take the scores that are zero's index to get a list of all users who have no ratings.
    no_ratings_grouped = df.groupby('user_id')['score'].agg(sum).to_frame()
    no_score_users = no_ratings_grouped[no_ratings_grouped['score'] == 0].index 

    # Group by user ID and apply a counter to each category, then transform it back into a normal dataframe.
    user_categories = subset.groupby("user_id")['category'].apply(Counter).to_frame().unstack(level=1, fill_value=0)
    user_categories.columns = user_categories.columns.droplevel(0)
    user_categories = user_categories.reset_index().fillna(0)

    # Group by user ID and apply a counter to each sub category, then transform it back into a normal dataframe. 
    user_sub_categories = subset.groupby("user_id")['sub_category'].apply(Counter).to_frame().unstack(level=1, fill_value=0)
    user_sub_categories.columns = user_sub_categories.columns.droplevel(0)
    user_sub_categories = user_sub_categories.reset_index().fillna(0)

    # Initialize a list of duplicate columns found in user_sub_categories to drop.
    drop_cols = ['user_id', 'sports', 'video', 'games', 'lifestyle', 'tv', 'news']
    user_sub_categories.drop(columns=drop_cols, inplace = True)

    # Create a complete user profile data frame by concatenating the two dataframes containing user preferences.
    complete_profile = pd.concat([user_categories, user_sub_categories], axis=1)

    # Create a list of nested lists to build new rows in the user preferences dataframe for users without any ratings. 
    data = [[user_id] + [0 for _ in range(len(complete_profile.columns)-1)] for user_id in no_score_users]

    # Concatenate the result on to finish obtaining user category and sub category preferences. 
    complete_profile = pd.concat([complete_profile, pd.DataFrame(data=data, columns=complete_profile.columns)]).sort_values('user_id')

    return complete_profile,  user_categories.columns, user_sub_categories.columns

def create_user_features(df):
    """
    Applies transformations to the dataset to create a user feature dataset as well as prepare data for clustering.

    Args:
        df (pd.DataFrame) : The full tensorflow dataset that gets modified.

    Returns:
        user (pd.DataFrame) : The user feature dataset generated by `apply_transformations`.
    """
    # Modify the tensorflow dataset with create_user_taste_profile to get user preferences.
    user, cat_columns, sub_cat_columns = create_user_taste_profile(df)
    
    # Scale the category preferences.
    category_sums = user[cat_columns[1:]].sum(axis=1)
    sub_category_sums = user[sub_cat_columns].sum(axis=1)
    user[cat_columns[1:]] = user[cat_columns[1:]].div(category_sums, axis=0).fillna(0)
    user[sub_cat_columns] = user[sub_cat_columns].div(sub_category_sums, axis=0).fillna(0)

    # Obtain the median hour of interaction for users and append it to the user feature matrix.
    medians = df.groupby("user_id")["time"].apply(np.median).reset_index()
    user["median_time"] = medians["time"]

    # Scale the median interaction time.
    user["median_time"] = user["median_time"] / 24
    return user

def create_item_features():
    """
    Creates item features for use in ALS and SGD. The resulting data is output to a csv and stored for later concatenation of cluster labels
    and eventual use for matrix factorization based recommender systems. 
    """

    # Read in the new with popularity dataset to have access to popularity counts.
    news = pd.read_csv("../MIND_large/csv/news_with_popularity.csv", index_col=0)

    # Create the popularity column in the item feature dataset by combining the history and impression popularity of an article.
    news['popularity'] =  news['popularity_history'] + news['popularity_impression']

    # Drop the history and impression popularity columns.
    news.drop(columns = ['popularity_history', 'popularity_impression', 'url', 'title_entities', 'abstract_entities'], inplace=True)

    # Dummy code the category variables sub_category and category, then concatenate the result to the item features.
    dummy_coded_categories = pd.get_dummies(news.drop(columns=['popularity', 'news_id', 'title', 'abstract']), dtype='float', prefix_sep="<sep>")
    drop_cols = ['sub_category<sep>' + col for col in ['travel', 'sports', 'video', 'games', 'lifestyle', 'tv', 'news']]
    dummy_coded_categories.drop(columns=drop_cols, inplace=True)
    dummy_coded_categories.columns = [column.split('<sep>')[-1] for column in dummy_coded_categories.columns]

    # Remove columns that are not included in user features, meaning in our dataset no user has interacted with them. 
    dummy_coded_categories = dummy_coded_categories.drop(columns=['autoslosangeles', 'autosmidsize','causes-poverty','finance-startinvesting','internationaltravel',
                                        'lifestylewhatshot','newsother','relationships','traveltrivia','tv-golden-globes-video','ustravel']) 

    item_features = pd.concat([news.drop(columns=['category', 'sub_category']), dummy_coded_categories], axis=1)

    # Return the item features.
    return item_features    

def chunk_tf_dataset(tf_dataset):
    """
    Breaks the Tensorflow compatible dataset into chunks.

    Args:
        tf_dataset (pd.DataFrame) : The Tensorflow compatible dataset to save as several chunks.
    """
    fpath = f'../MIND_large/csv/tensorflow_dataset'
    idx = [5161473 * i for i in range(5)]
    idx[-1] += 1
    for i in range(len(idx)-1):
        start, end = idx[i], idx[i+1]
        chunk = tf_dataset[(tf_dataset.index >= start) & (tf_dataset.index < end)]
        chunk.to_csv(fpath + f'_chunk{i}.csv')

def check_duplicates(df):
    """
    deprecated, tensorflow compatible dataset now contains no duplicates of histories
    Could be utilized to check if duplicates still exist within the dataset
    """
    timeless = df.drop("time", axis=1)
    dups = timeless[timeless.duplicated].index
    new_df = df.drop(index=dups, inplace=True)
    print(f"Removed {len(dups)} duplicates.")

def add_embeddings():
    """
    Something to add embeddings to item features, potential use but other stuff to do before this
    """
    # if umap_dim != 0:

    #     # Subset for only the text data of the news dataset.
    #     news_text = news.drop(columns=['news_id', 'category', 'sub_category', 'popularity'])
        
    #     # Use clustering modules to vectorize and then create reduced embeddings of the text for the best performing parameters.
    #     _, tf_matrix = cm.vectorize_items(news_text)
    #     reduced_embeddings = cm.create_UMAP_embeddings(umap_dim, tf_matrix)

    #     for i in range(umap_dim):
    #         news[f'reduced_embeddings_{i+1}'] = reduced_embeddings[:, i]
