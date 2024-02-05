import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
"""
Contains modules for exploratory data analysis and transformations specific to the microsoft MIND dataset.
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
def data_to_csv(fpath='../MIND_small/tsv/behaviors.tsv', behaviors=True):
    """
    Takes a tab seperated variable file from the MIND dataset, adds columns to it, and exports it as a CSV.

    Args:
        fpath (str) : The path to the directory that the csv will be output to.

    Returns:
        None
    """
    if behaviors:
        behaviors_columns = ['impression_id', 'user_id', 'time', 'history', 'impressions']
        df = pd.read_csv(fpath, sep='\t', names=behaviors_columns)
        df.to_csv('MIND_small/csv/behaviors.csv')
        
    else:
        news_columns = ['news_id', 'category', 'sub_category', 'title', 'abstract', 'url', 'title_entities', 'abstract_entities']
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

def check_distributions(dataframe):
    ## maybe, could be totally useless now that I think about it hahahaha
    """
    Takes a dataframe, melts it and then plots it so that the distributions of each feature can be examined.

    Args:
        dataframe (pd.DataFrame) : A dataframe containing features to visualize.

    Returns:
        feature_distributions (sns.FacetGrid) : A SNS facet grid containing charts with the features visualizations.    
    """

    # Initialize the dataframe
    df = dataframe.copy()

    # Get the features for extraction
    features = df.columns

    # Melt the data so that distributions can be examined
    df_long = df.melt(id_vars=['genre', 'subgenre'], value_vars=features, var_name='feature', value_name='value')

    # Create a seaborn facetGrid object with the metled dataframe to house all distribution graphs
    feature_distributions = sns.FacetGrid(df_long, col="feature", hue="genre", palette="rocket_r", col_wrap=3, sharex=False, sharey=False)
    feature_distributions = (feature_distributions.map(sns.histplot, "value", element="step").add_legend())

    # Create titles for the distributions based off of the column names
    feature_distributions.set_titles("{col_name}")

    # Establish a tight layout
    feature_distributions.fig.tight_layout()

    # Return the chart object for use in the jupyterNotebook
    return feature_distributions

# features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:

def plot_categories(dataframe: pd.DataFrame) -> sns.FacetGrid:
    """
    Creates a data visualization to explore the distribution of categories and sub-categories.

    Args:
        dataframe (pd.DataFrame) : A dataframe from which genres and subgenres can be extracted.  

    Returns:
        categories_distribution (sns.FacetGrid) : A sns plot object containing the distributions for genres and subgenre 
    """
    
    # Initialize a list contianing the names of the categories we will be examining
    category_cols = ['category','sub_category']

    # Utilize dataframe methods to create a dataframe that is counts of each category
    category_data = dataframe.group_by(category_cols).agg('count')
    print(category_data)
    




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