from datetime import datetime
from collections import Counter
import os
from transformers import BertTokenizer, TFBertModel
import keras.api._v2.keras
import tensorflow as tf
import pandas as pd
import numpy as np
import umap.umap_ as umap
import sklearn.cluster as cluster


"""
Data processing is done in this module in order to avoid slowdown from hardware constraints in the eda_report.
"""

def data_to_csv(behaviors : bool, fpath : str = '../MIND_small/tsv/behaviors.tsv') -> None:
    """
    Takes a tab seperated variable file from the MIND dataset, adds columns to it, and exports it as a CSV.

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

    # Split the impression by '-'
    impression_info = impression.split('-')

    # Return a dictionary with the articleID and the click indicator
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

    # Iterate through every row of the columns history and impressions in the dataframe.
    for user_id, history, impressions in zip(behaviors_frame['user_id'], behaviors_frame['history'], behaviors_frame['impressions']):

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

def create_popularity_csvs(news : pd.DataFrame, behaviors : pd.DataFrame, small : bool=True) -> None:
    """
    Extracts popularity for categories and articles from user history and impressions then exports them to csvs.
    This is done by using the eda_modules.create_popularity_dfs which creates four separate dataframes which represent article
    popularity and category popularity. The article and category pairs are then concatenated and exported for usage within
    the exploratory data analysis report.
    
    Args:
        news (pd.DataFrame) : The dataframe containing the data from the news csv.
        behaviors (pd.DataFrame) : The dataframe containing the data from the behaviors csv.
    
    Returns:
        None
    """

    # Create the dataframes with information about how populary each category is.
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

    # Concatenate the 2 dataframes on rows
    categories = pd.concat([cat_test, cat_test2], axis=0)
    
    # Output the CSVs to the respective folder
    if small:
        news.to_csv("../MIND_small/csv/news_with_popularity.csv")
        categories.to_csv('../MIND_small/csv/category_with_popularity.csv')
    else:
        news.to_csv("../MIND_large/csv/news_with_popularity.csv")
        categories.to_csv('../MIND_large/csv/category_with_popularity.csv')
from tqdm import tqdm
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
    
    # Creating an update_data function to better manage boilerplate.
    def update_data(data : dict, user_id : str, time_stamp : str, news_id : str, interaction_type : str, score : int) -> dict:
        """
        Updates the data dictionary with a users interaciton data.
        """
        # Updating all relevant keys in the dictionary
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

    # Initializing a counter so that the number of new rows can be controlled for easier testing and processing.
    seen = []
    # Iterating through all relevant information.
    print('Starting loop')
    for user_id, history, impressions, time_stamp in tqdm(zip(behaviors['user_id'], behaviors['history'], behaviors['impressions'], behaviors['hour']), total=len(behaviors['user_id'])):

        # Iterating through all news ids in the history and impressions.
        if user_id not in seen:
            if type(history) != float:
                for news_id in history.split():
                    data = update_data(data, user_id, time_stamp, news_id, 'history', 1)

                seen.append(user_id)
        for impression in impressions.split():
            impression_result = clean_impression(impression)
            if impression_result['score'] == 1:
                data = update_data(data, user_id, time_stamp, impression_result['article_ID'], 'impression', 1)   

            else:
                data = update_data(data, user_id, time_stamp, impression_result['article_ID'], 'impression', 0)   

    # Returning a dataframe with the dictionary as its input.
    return pd.DataFrame(data=data)

def create_text_embeddings(news):
    """
    Applies pre-trained BERT embeddings to the feature columns with text data for use within clustering methods.

    Args:
        news (pd.DataFrame) : News dataset containing the articles with different abstracts and title for embedding collection.

    Returns:
        dataset (pd.DataFrame) : A dataset with embeddings columns.
    """
    # Initialize the bert tokenizer and model from bert base cased.
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    model = TFBertModel.from_pretrained('bert-base-cased')

    # Create a dense layer with the embedding dimension to process embeddings.
    embedding_dimension = 8
    dense_layer = keras.layers.Dense(embedding_dimension, activation='linear')

    # Define a function to get the embeddings from the models and apply them to the text.
    def get_embeddings(text_1):
        """
        Gets embeddings from pre trained bert model for news information used for clustering.
        """
        try:
            if type(text_1) == float:
                return [0]

            # Apply the tokenizer to the text and return it in the tensorflow tensor format. 
            encoded_text = tokenizer(text_1, return_tensors='tf')

            # Get the output from BERT model with the encoded text.
            bert_output = model(encoded_text)

            # Use the pooled output for a single vector representation of the input. 
            pooled_output = bert_output.pooler_output

            # Apply dense layer to project to desired size.
            # embedding_vector = list(dense_layer(pooled_output).numpy())
            # return embedding_vector
            return pooled_output.numpy().tolist()
        except:
            print(text_1)

    # If to be implemented further data collation needs to be utilized 
    # doesnt tokenzier do that?
    # example text that doesnt work From Tiger Woods' historic win in Japan to a major shake-up at CBS Sports, here is what you missed from golf this weekend.
    # Real talk. Demi Moore got candid about a variety of topics in her new book, Inside Out, including her famous exes, substance abuse struggles and her heartbreaking sexual assault. "The same question kept going through my head: How did I get here?" the 56-year-old actress began in the memoir, which was released on Tuesday, September 24. "The husband who I'd thought was the love of my life had cheated on me and then decided he didn't want to work on our marriage. My children weren't speaking me. … Is this life? I wondered. Because if this is it, I'm done." Moore provided insight into all three of her marriages in the book. She was married to Freddy Moore from 1980 to 1985, Bruce Willis from 1987 to 2000 and Ashton Kutcher from 2005 to 2013. The end of the G.I. Jane star's relationship with the former That 70's Show star, however, seemed to have the biggest impact on her. "I lost me," the Ghost actress told Diane Sawyer on Good Morning America on Monday, September 23, about their split. "I think the thing if I were to look back, I would say I blinded myself and I lost myself." Moore and Kutcher, who is 15 years her junior, started dating in 2003. After Us Weekly broke the news that he was allegedly unfaithful in 2011, the twosome called it quits. The Ranch star married Mila Kunis in July 2015. They share two kids: Wyatt, 4, and Dimitri, 2. Kutcher, for his part, reflected on the divorce during an appearance on Dax Shepard's "Armchair Expert" podcast last year. "Right after I got divorced, I went to the mountains for a week by myself," Kutcher told Shepard in February 2018. "I did no food, no drink   just water and tea. I took all my computers away, my phone, my everything. I was there by myself, so there was no talking. I just had a notepad, a pen and water and tea   for a week." He referred to the trip as "really spiritual and kind of awesome." "I wrote down every single relationship that I had where I felt like there was some grudge or some anything, regret, anything," Kutcher explained. "And I wrote letters to every single person, and on day seven, I typed them all out and then sent them." While Moore certainly doesn't hold back in Inside Out, a source told Us earlier this month that the Kutcher isn't worried about the book. "Ashton knew what was coming. He had a heads up on what is in the book," the insider said on September 13. "He's not mad or disappointed. This is Demi's truth, and he always felt sympathetic toward her. He knows her story and that her upbringing was difficult." Inside Out is available now. Scroll through for 10 revelations from the book:
    
    news['abstract_embeddings'] = news['abstract'].apply(get_embeddings)
    news['title_embeddings'] = news['title'].apply(get_embeddings)
    return news

def create_interaction_counts():
    """
    Creates interaction counts dataframes for usage in timestamp analysis.
    """
    
    # Load in the datasets and set the index of the news dataset to news id    
    news = pd.read_csv('../MIND_large/csv/news.csv')
    copynews = news.set_index('news_id')
    behaviors = pd.read_csv('../MIND_large/csv/behaviors.csv')
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

    print('starting history popularity')
    behaviors[category_popularity.columns.to_list()[1:]] = behaviors.apply(lambda row : get_interaction_popularity(row, True), axis='columns', result_type='expand')
    behaviors.rename(columns={column : column + "_history" for column in category_popularity.columns.to_list()}, inplace=True)
    print('starting impression popularity')
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

def simple_string_to_list(input_str):
    """Reformats separate list of embeddings so it is compatible for expansion"""    
    processed_string = input_str.replace("[[", "[").replace("]]", "]").replace('\n', '').replace(', dtype=float32)]', '')
    array = list(eval(processed_string))
    return array

def preprocess_BERT_embeddings(news : pd.DataFrame, small : bool) -> None:
    """
    Prepares and applies BERT embeddings to the dataset for usage within clustering.  
    """
    # news = create_text_embeddings(news)
    if small:
        fpath = '../MIND_small'
    else:
        fpath = '../MIND_large'

    # news.to_csv(fpath + '/csv/news_BERT_embeddings.csv')
    # del news
    print('starting')
    embedded_news = pd.read_csv(fpath + '/csv/news_big_embeddings.csv', index_col = 0).drop(columns=['abstract_entities', 'title_entities', 'url'])
    embedded_news = embedded_news[embedded_news['abstract_embeddings'] != '[0]']
    embedded_news = embedded_news[embedded_news['abstract_embeddings'].isna() == False]
    embedded_news['abstract_embeddings'] = embedded_news['abstract_embeddings'].apply(lambda x : simple_string_to_list(x))
    embedded_news['title_embeddings'] = embedded_news['title_embeddings'].apply(lambda x : simple_string_to_list(x))
    abstracts = pd.DataFrame(embedded_news['abstract_embeddings'].to_list(), index=embedded_news.index)
    titles = pd.DataFrame(embedded_news['title_embeddings'].to_list(), index=embedded_news.index)
    titles.columns = ['{}_title'.format(title) for title in titles.columns]
    abstracts.columns = ['{}_abstract'.format(title) for title in abstracts.columns]
    embedded_news = pd.concat([embedded_news, titles, abstracts], axis=1).drop(columns=['abstract_embeddings','title_embeddings', 'title', 'abstract'])
    embedded_news.to_csv(fpath + '/csv/news_BERT_extracted_embeddings.csv')

def create_hourly_long():
    """
    Creates long format dataframes with hourly popularity counts for data visualizations and as a potential feature for popularity.

    Args:
        None
    
    Returns:
        behaviors2, unique_user_histories2 (pd.DataFrame) : Two pandas dataframes containing long format popularity counts for later graphing.
    """
    # Load in the behaviors with individual counts dataset and the news dataset.
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
    Uses the tensorflow compatible dataset to create the user feature dataframe.

    Args:
        df (pd.DataFrame) : The tensorflow compatible dataset as a dataframe for modification.

    Returns:
        users (pd.DataFrame) : The user feature dataframe.
    """
    # Subset the dataframe for items where a user has a positive rating and then group by user ids applying a counter.
    user_prof = df.groupby("user_id")['category'].apply(Counter).to_frame()

    # Unstacks the new columns and then removes multi-indexing.
    user = user_prof.unstack(level=1, fill_value=0)
    user.columns = user.columns.droplevel(0)
    user = user.reset_index()

    # Fill NaN values with zero and return the user frame.
    user = user.fillna(0) 
    return user

def convert_time(dates):
    """
    Modifies the list of datetime strings in a user row into datetime objects for use within feature extraction.

    Args:
        dates (list) : List of strings containing date and time of interaction.

    Returns:
        time_objs (list) : List of datetime objects corresponding to each date string in the list given.
    """
    # Initialize an empty list to store datetime objects.
    time_objs = []

    # Set up the regular expression to use in datetime.
    time_regex = '%m/%d/%Y %I:%M:%S %p'

    # For all dates in the users dates transform the date into a datetime object and add it to time_objs.
    for date in dates:
        time_objs.append(datetime.strptime(date,time_regex))

    # Return time_objs.
    return time_objs

def create_times(df):
    """
    Creates a dataframe of users and a list of all their interaction time stamps as datetime objects. 

    Args:
        df (pd.DataFrame) : The tensorflow compatible dataset as a dataframe for modification.

    Returns:
        date_objs (pd.DataFrame) : Dataframe containing users and a list of datetime objects of their interaction time stamps.
    """
    
    # Group by user id and apply a list to each users time columns.
    date_strings = df.groupby("user_id")["time"].apply(list)
    
    # Apply the convert time function and return the resulting dataframe.
    date_objs = date_strings.map(convert_time)
    return date_objs

def median_hour(dates):
    """
    Finds the median hour that users interacted with articles.

    Args:
        dates (list) : List of datetime objs.

    returns:
        (int) : The median hour of their interactions.
    """
    # Initialize an empty list to hold the hours of a users interactions.
    hours = []

    # Populate the list with hours
    for date in dates:
        hours.append(date.hour)
    
    # Return the median hour of their interactions.
    return np.median(hours)

def remove_dups_tf (df):
    """
    deprecated, tensorflow compatible dataset now contains no duplicates of histories
    Could be utilized to check if duplicates still exist within the dataset
    """
    timeless = df.drop("time", axis=1)
    dups = timeless[timeless.duplicated].index
    new_df = df.drop(index=dups, inplace=True)
    print(f"Removed {len(dups)} duplicates.")

def apply_mean_scale (row):
    """
    Applies a mean scale to the user preferences.

    Args:
        row (pd.DataFrame) : A row from the user features matrix to apply scaling to.

    Returns:
        Returns the values in the row after having been scaled.
    """
    # Get the total of all values in the row
    total = row.values.sum()

    # Initialize and then populate a list of all values after having been scaled.
    new_values = []
    for value in row:
        value = value / total
        new_values.append(value)

    # Return the values as a series
    return pd.Series(new_values)
        
def scaling_data(df):
    """
    Scaling data applies mean scaling to the users features as a method of normalization utilizing the apply_mean_scale function defined above.

    Args:
        df (pd.DataFrame) : The user feature dataframe containing popularity counts for each category.

    Returns:
        df (pd.DataFrame) : The user mean scaled user feature matrix.
    """
    # Modifies all columns containing category counts to have been mean scaled.
    df.iloc[:,1:-1] = df.loc[:, (df.columns != 'user_id') & (df.columns != 'median')].apply(lambda x: apply_mean_scale(x), axis=1)

    # Return the scaled dataframe.
    return df

def create_user_taste_profile_sub(df):
    """
    Uses the tensorflow compatible dataset to create the user feature for sub category preferences dataframe.

    Args:
        df (pd.DataFrame) : The tensorflow compatible dataset as a dataframe for modification.

    Returns:
        users (pd.DataFrame) : The user feature dataframe.
    """
    # Subset the dataframe for items where a user has a positive rating and then group by user ids applying a counter.
    history = df[df['score'] == 1]
    user_prof = history.groupby("user_id")['sub_category'].apply(Counter).to_frame()

    # Unstacks the new columns and then removes multi-indexing.
    user = user_prof.unstack(level=1, fill_value=0)
    user.columns = user.columns.droplevel(0)
    user = user.reset_index()

    # Fill NaN values with zero and return the user frame.
    user = user.fillna(0)
    return user

def apply_transformations(df):
    """
    Applies transformations to the dataset to create a user feature dataset as well as prepare data for clustering.

    Args:
        df (pd.DataFrame) : The full tensorflow dataset that gets modified.

    Returns:
        user (pd.DataFrame) : The user feature dataset generated by `apply_transformations`.
    """
    user = create_user_taste_profile(df)
    
    # Obtain the median hour of interaction for users and append it to the user feature matrix.
    medians = df.groupby("user_id")["time"].apply(np.median).reset_index()
    user["median"] = medians["time"]

    # Scale the category preferences.
    user = scaling_data(user)

    # Scale the median interaction time.
    user["median"] = user["median"] / 24
    return user
    
def load_dataset():
    full = pd.DataFrame()
    for i in range(4):
        df = pd.read_csv(f"../MIND_large/csv/tensorflow_dataset_chunk{i}.csv", index_col=0)
        full = pd.concat([full, df])
    return full
