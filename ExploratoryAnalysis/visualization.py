from numpy import mat
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.axes
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import os

# Path handling
module_dir = os.path.dirname(__file__) 
data_path = os.path.join(module_dir, '../MIND_large/csv')

"""
Contains functions for data visualizations in the exploratory data analysis report and maybe more? 
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
    category_popularity = pd.read_csv(data_path + '/category_with_popularity.csv').drop(columns=['Unnamed: 0'])
    category_popularity_long = category_popularity.melt(id_vars='popularity_type', value_vars=category_popularity.columns, value_name='popularity')
    category_popularity_long_sorted = category_popularity_long.sort_values(by=['popularity_type', 'popularity'], ascending=[True, False])

    # Create a seaborn FacetGrid for the visualization.
    g = sns.FacetGrid(data = category_popularity_long_sorted, col='popularity_type', sharex=False, height=5, aspect=1, hue='variable')

    # Apply a barplot to each facet and set the labels and titles.
    g.map(sns.barplot, 'popularity','variable')
    g.set_axis_labels(x_var="Count of interactions", y_var='Categories')
    g.set_titles("User {col_name}")
    g.add_legend(title='Categories')
    plt.savefig(data_path+"/cat_pop.png")
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
    article_popularity = pd.read_csv(data_path + '/news_with_popularity.csv', index_col=0)
    article_popularity['total'] = article_popularity['popularity_impression'] + article_popularity['popularity_history']
    article_popularity = article_popularity.sort_values(by='total', ascending=False).reset_index(drop=True)

    g = sns.lineplot(data=article_popularity['total'])
    plt.xlabel('Articles')
    plt.ylabel('Count of interactions')
    plt.savefig(data_path+"/art_pop.png")
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

    g = sns.FacetGrid(behaviors_long, col='type', sharey=False, hue='variable', height=5, aspect=1, palette='tab20')

    g.map(sns.lineplot, 'hour', 'value')
    g.set_axis_labels(y_var='Popularity', x_var='Time of day')
    g.add_legend(title='Categories')
    g.set_titles('User {col_name}')
    plt.show()

def plot_both():
    fig,ax = plt.subplots(1,3)
    ax = ax.flatten()

    plot_category_popularity(ax=ax)
    plot_article_popularity(ax=ax[2])
