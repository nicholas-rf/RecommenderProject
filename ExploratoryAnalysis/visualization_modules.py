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
                    path=['category', 'sub_category'],  
                    values='number_of_articles',       
                    title='Categories and their sub-categories')
    
    # Return the figure so adjustments can be made inside of the notebook for experimentation.
    return fig  

def plot_category_popularity():
    """
    Plots category popularity for user impressions and history separately using an SNS FacetGrid.
    
    Args:
        None
    
    Returns:
        None
    """
    # Melt and then sort the data.
    category_popularity = pd.read_csv('../MIND_large/csv/category_with_popularity.csv').drop(columns=['Unnamed: 0'])
    category_popularity_long = category_popularity.melt(id_vars='popularity_type', value_vars=category_popularity.columns, value_name='popularity')
    category_popularity_long_sorted = category_popularity_long.sort_values(by=['popularity_type', 'popularity'], ascending=[True, False])
    
    # Create a seaborn FacetGrid for the visualization.
    g = sns.FacetGrid(category_popularity_long_sorted, col='popularity_type', sharex=False, height=5, aspect=1, hue='variable')

    # Apply a barplot to each facet and set the labels and titles.
    g.map(sns.barplot, 'popularity','variable')
    g.set_axis_labels(x_var="Count of interactions", y_var='Categories')
    g.set_titles("User {col_name}")
    g.add_legend(title='Categories')

    plt.show()

def plot_article_popularity():
    """
    Plots category popularity for the catalogue to determine if some articles have more ratings than others.
    
    Args:
        article_popularity (pd.DataFrame) : A pandas dataframe containing the popularity of each category within impressions and user history's.
    
    Returns:
        None
    """
    # Melt and then sort the data.
    article_popularity = pd.read_csv('../MIND_large/csv/news_with_popularity.csv', index_col=0)
    article_popularity['total'] = article_popularity['popularity_impression'] + article_popularity['popularity_history']
    article_popularity = article_popularity.sort_values(by='total', ascending=False).reset_index(drop=True)
    g = sns.lineplot(data=article_popularity['total'])
    plt.xlabel('Articles')
    plt.ylabel('Count of interactions')
    plt.show()


def create_temporal_graphs(behaviors_with_counts : pd.DataFrame, history_counts) -> None:
    """ 
    Creates graphs showcasing popularity of certain categories for different times of day with a seaborn facet grid.

    Args:
        behaviors_with_counts (pd.DataFrame) : A dataframe containing the hourly counts of popularity.

    Returns:
        None 
    """

    cols = ['lifestyle', 'health', 'news', 'sports', 'weather', 'entertainment', 'autos', 'travel', 'foodanddrink', 'tv', 'finance', 'movies', 'video', 'music', 'kids', 'middleeast']
    impression_ = []
    history_ = []
    for col in cols:
        impression_.append(col + '_impression')
        history_.append(col + '_history')

    behaviors_long = behaviors_with_counts.melt(id_vars='hour', value_vars=impression_)
    history_long = history_counts.melt(id_vars='hour', value_vars=history_ )
    behaviors_long['type'] = behaviors_long['variable'].apply(lambda x : x.split('_')[1])
    behaviors_long['variable'] = behaviors_long['variable'].apply(lambda x : x.split('_')[0])
    history_long['type'] = history_long['variable'].apply(lambda x : x.split('_')[1])
    history_long['variable'] = history_long['variable'].apply(lambda x : x.split('_')[0])

    behaviors_long = pd.concat([behaviors_long, history_long], axis=0)

    g = sns.FacetGrid(behaviors_long, col='type', sharey=False, hue='variable', height=5, aspect=1)

    g.map(sns.lineplot, 'hour', 'value')
    g.set_axis_labels(y_var='Popularity', x_var='Time of day')
    g.add_legend(title='Categories')
    g.set_titles('User {col_name}')
    plt.show()


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