from numpy import mat
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.axes
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

"""
Contains modules for exploratory data analysis and transformations specific to the microsoft MIND dataset.
"""

### Introductory data processing ### 

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


### News Dataset Methods ### 

def plot_categories(news: pd.DataFrame) -> matplotlib.axes.Axes:
    """
    Creates a sns countplot for all categories to show dominant categories.

    Args: 
        news (pd.DataFrame) : The news dataframe from which categories get plotted.
    
    Returns:
        fig (sns.countplot) : An sns.countplot containing the visualization for genres.
    """
    
    # Create a count plot for the category on the y axis so that all category names can fit into the chart.
    fig = sns.countplot(news, y='category', hue='category')
    return fig


def plot_sub_categories(news : pd.DataFrame) -> go.Figure:
    """
    Creates a data visualization to explore the distribution of categories and sub-categories.

    Args:
        news (pd.DataFrame) : The news dataframe from which categories and sub-categories can be extracted.  

    Returns:
        fig (go.Figure) : A plotly treemap plot containing the visualization for genres and subgenre 
    """
    
    # Initialize a list contianing the category and sub_category columns.
    category_cols = ['category','sub_category']

    # Group by the categories and count the total number of articles in each one.
    sub_category_data = news.groupby(category_cols).agg(number_of_articles=('news_id','count'))

    # Reset the index to make the data into long-format.
    sub_category_data.reset_index(inplace=True)

    # Set up a plotly treemap with the category data determining size of blocks by the number of articles.
    fig = px.treemap(sub_category_data, 
                    path=['category', 'sub_category'],  # Hierarchical structure
                    values='number_of_articles',            # Size based on Value1
                    title='Categories and their sub-categories')
    
    # Return the figure so adjustments can be made inside of the notebook for experimentation.
    return fig
    
def missing_news_analysis(news : pd.DataFrame):
    """
    Creates a chart for missing values in the news dataset to determine missingness.

    Args:
        news (pd.DataFrame) : The news dataset to examine missing values from.

    Returns:
        fig (some figure type) : something
    """
    missing_vals = news.isna()
    missing_vals['abstract'] # can utilize this as a mask to get values where missing abstract is true in the main, to see if theres a
    # majority of missing abstracts for a specific category 


### behaviors dataset methods ###
    
    ## important things to do
    ## plot distributions of time and interactions to gauge popular interaction times
    ## analyze the average clickthrough rates of articles per genre (ie total reccommendations given and clickthrough rates ) 


def create_popularity_df(news_frame : pd.DataFrame, behaviors_frame : pd.DataFrame) -> pd.DataFrame:
    """ 
    Creates a dataframe with the popularity of every sub-category and category. The category popularity gets calculated as a biproduct of the sub-category popularity.
    
    Args:
        news_frame (pd.DataFrame) : The news dataframe so that the index can be set to the news ID and used when querying user history.
        behaviors_frame (pd.DataFrame) : The behaviors dataframe so that users last interaction history can be used to understand category popularity.
    
    Returns:
        category_popularity (pd.DataFrame) : A dataframe with categories as columns and popularity as a row.
    """
    
    max_idcs = behaviors.groupby('user_id')['time'].idxmax()
    copynews = news_frame.set_index('news_id')
    max_behaviors = behaviors_frame.loc[max_idcs]
    popoularity_dict = {category: 0 for category in pd.unique(copynews['category'])}
    for history in max_behaviors['history']:
        if type(history) != float:
            for news_id in history.split():
                sub_category = copynews.loc[news_id]['category']
                popoularity_dict[sub_category] += 1

    return pd.DataFrame(data=popoularity_dict, index=[1])

def check_genre_popularity():
    """
    
    """


def check_temporal_clicks(dataframe):
    # plots clickthrough rates throughout the day to analyze the affect that time has on clickthrough rates
    """
    Creates a data visualization of clickthrough rates throughout the day to analyze the affect that time has on clickthrough rates.

    Args:
        dataframe (pd.DataFrame) : A dataframe from which genres and subgenres can be extracted.  

    Returns:
        temporal_chart (sns.something) : A sns plot object containing the number of clicks for a given time?
    """
    # maybe want to group by time and then sum the amount of clicks for each time and do a simple line chart 


def check_click_diversity():
    # Checks the click diversity of news articles in order to determine potential for personalization
    """
    
    """

### POTENTIAL TO SCRAPE ARTICLE INFORMATION FOR SOME SORT OF TEXT ANALYSIS WITH CHATGPT API OR WITH AN LLM LIKE T5 ### 
### GOING TO WANT A SEPERATE MODULE FOR PREPROCESSING STEPS ### 

def check_tail_genre(dataframe):
    # TODO check the typing of the return value for the function
    # what sorts of long tails can be exhibited by our data here? 
    # There could be an abundance of user ratings for specific genres or specific news articles themselves, solution could be two functions for checking each?
    # also temporal column could be being utilized in some way by the data in an annoying way

    """
    Creates a data visualization to examine the amount of ratings genres recieve so that the long tail problem can be avoided.

    Args:
        dataframe (pd.DataFrame) : A dataframe which will be charted to see if an IDF needs to be applied to ratings.

    Returns:
        genre_tail_check (?sns.plot?) : A plot object to determine if a long tail is exhibited in the data. 
    """
    # Copy the dataframe
    df = dataframe.copy()

    # Check the articles    




def check_tail_articles(dataframe):
    """
    Creates a data visualization to examine the amount of ratings articles recieve so that the long tail problem can be avoided.

    Args:
        dataframe (pd.DataFrame) : A dataframe which will be charted to see if an IDF needs to be applied to ratings.

    Returns:
        article_tail_check (?sns.plot?) : A plot object to determine if a long tail is exhibited in the data. 
    """



## What do we want to do for our EDA with the MIND dataset? ## 

##  Want to analyze the tail of reviews, to check if we want to implement an IDF weight to improve coverage
##  Want to determine counts of the genres of all reviews
##  Want to evaluate how many unique users there are, and the most popular interactions

##  maybe want to create a total impressions column for news articles based off of users interaction history 
##  distributions of categories and subcategories (maybe heatmaps)

## all unique user ids? 

## maybe feature extraction for genres?

## want to keep boilerplate minimal in jupyter book so will define functions to be called here
## Starting with basic EDA

## Everything in this file is rough draft as of now as it has yet to be tested inside of a notebook, however that is fine