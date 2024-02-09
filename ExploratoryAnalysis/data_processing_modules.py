import ExploratoryAnalysis.visualization_modules as visualization_modules
import pandas as pd
import numpy as np

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
        df.to_csv('MIND_small/csv/behaviors.csv')

    # If we are are not changing the format of the behaviors tsv, use the news csv.
    else:

        # Column names to be added.
        news_columns = ['news_id', 'category', 'sub_category', 'title', 'abstract', 'url', 'title_entities', 'abstract_entities']

        # Read in the tsv adding names and then export it to a csv.
        df = pd.read_csv(fpath, sep='\t', names=news_columns)
        df.to_csv('MIND_small/csv/news.csv')

def check_data_types(dataframe):
    """
    Prints out data types of a pandas dataframe in a clean way for the EDA notebook.
    
    Args:
        dataframe (pd.DataFrame) : A dataframe to extract variable types from.

    Returns:
        datatypes (pd.DataFrame) : A dataframe containing datatypes as a row and fetures as columns.
    """

    return pd.DataFrame(data=dataframe.dtype.tolist(), columns=dataframe.columns)


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

def create_popularity_dfs(news_frame : pd.DataFrame, behaviors_frame : pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
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
    
    # Done for getting last history of every user (can be used for popularity up to a time when training so keeping here)
    # max_idcs = behaviors_frame.groupby('user_id')['time'].idxmax()
    # max_behaviors = behaviors_frame.loc[max_idcs]

    # Create a copy of the news dataframe where the index is the news_id.
    copynews = news_frame.set_index('news_id')

    # Create empty dictionaries that get filled while iterating through the dataset.
    article_popularity_impression = {article: 0 for article in news_frame['news_id']}
    article_popularity_history = {article: 0 for article in news_frame['news_id']}
    category_popularity_impression = {category: 0 for category in pd.unique(copynews['category'])}
    category_popularity_history = {category: 0 for category in pd.unique(copynews['category'])}

    # Iterate through every row of the columns history and impressions in the dataframe.
    for history, impressions in zip(behaviors_frame['history'], behaviors_frame['impressions']):

        # If our history is not a NaN.
        if type(history) != float:

            # Split the history into news IDs
            for news_id in history.split():

                # Locate the article and access its category, then increment the metrics.
                category = copynews.loc[news_id]['category']
                category_popularity_history[category] += 1
                article_popularity_history[news_id] += 1
                

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

def modify_time():
    """ 
    Modifies the time of the behaviors dataframe so that temporal analysis of popularity can occur. Can consider for collaborative
    filtering looking at users that are also interacting at a similar time  
    """