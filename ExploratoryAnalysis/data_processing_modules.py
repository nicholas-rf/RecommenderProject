from transformers import BertTokenizer, TFBertModel
import keras.api._v2.keras
import tensorflow as tf
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
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


def decompose_interactions(num_iterations : int, news : pd.DataFrame, behaviors : pd.DataFrame) -> pd.DataFrame:
    """
    Iterates through user interactions to create a tensorflow compatible dataset comprised of multiple rows per user, each row detailing
    their interaction, and relevant information about that interaction.
    
    Args:
        num_iterations (int) : The number of rows in the resulting dataframe.
        news (pd.DataFrame) : The dataframe containing the data from the news csv.
        behaviors (pd.DataFrame) : The dataframe containing the data from the behaviors csv.
    
    Returns:
        results (pd.DataFrame) : The dataframe containing users and their rankings. The columns of the dataframe are
        user_id, article_id, rating, rating_type, category, sub_category, timestamp. 
    """
    data = {'user_id' : [], 'time' : [], 'news_id' : [], 'category' : [], 'sub_category' : [], 'title' : [], 'abstract' : [], 'interaction_type' : [], 'score' : []}

    # Setting the index of news to news_id so that lookup for information can be done.
    news.set_index('news_id', inplace=True)

    # Creating an update_data function to better manage boilerplate.
    def update_data(data : dict, user_id : str, time_stamp : str, news_id : str, interaction_type : str, score : int) -> dict:
        """
        Updates the data dictionary with a users interaciton data.
        """
        # Updating all relevant keys in the dictionary
        data['user_id'].append(user_id)
        data['time'].append(time_stamp)
        data['news_id'].append(news_id)
        data['category'].append(news.loc[news_id]['category'])
        data['sub_category'].append(news.loc[news_id]['sub_category'])
        data['title'].append(news.loc[news_id]['title'])
        data['abstract'].append(news.loc[news_id]['abstract'])
        data['interaction_type'].append(interaction_type)
        data['score'].append(score)
        return data

    # Initializing a counter so that the number of new rows can be controlled for easier testing and processing.
    counter = 0

    # Iterating through all relevant information.
    for user_id, history, impressions, time_stamp in zip(behaviors['user_id'], behaviors['history'], behaviors['impressions'], behaviors['time']):
        if counter > num_iterations:
            break

        # Iterating through all news ids in the history and impressions.
        for news_id in history.split():
            data = update_data(data, user_id, time_stamp, news_id, 'history', 1)
            counter += 1
        for impression in impressions.split():
            impression_result = clean_impression(impression)
            if impression_result['score'] == 1:
                data = update_data(data, user_id, time_stamp, impression_result['article_ID'], 'impression', 1)   
                counter += 1
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

    # Apply the embeddings to the abstract and title columns.
    # abstracts = dataset['abstract'].to_list()
    # text = get_embeddings(abstracts[0])
            
    ## NEED TO INCLUDE TRUNCATING / COLLATING / SOMETHING LIKE THAT FOR THIS MODEL AS THERE ARE ABSTRACTS LARGER THAN INPUT LENGTH
            # doesnt tokenzier do that?
            # example text that doesnt work From Tiger Woods' historic win in Japan to a major shake-up at CBS Sports, here is what you missed from golf this weekend.
            # Real talk. Demi Moore got candid about a variety of topics in her new book, Inside Out, including her famous exes, substance abuse struggles and her heartbreaking sexual assault. "The same question kept going through my head: How did I get here?" the 56-year-old actress began in the memoir, which was released on Tuesday, September 24. "The husband who I'd thought was the love of my life had cheated on me and then decided he didn't want to work on our marriage. My children weren't speaking me. … Is this life? I wondered. Because if this is it, I'm done." Moore provided insight into all three of her marriages in the book. She was married to Freddy Moore from 1980 to 1985, Bruce Willis from 1987 to 2000 and Ashton Kutcher from 2005 to 2013. The end of the G.I. Jane star's relationship with the former That 70's Show star, however, seemed to have the biggest impact on her. "I lost me," the Ghost actress told Diane Sawyer on Good Morning America on Monday, September 23, about their split. "I think the thing if I were to look back, I would say I blinded myself and I lost myself." Moore and Kutcher, who is 15 years her junior, started dating in 2003. After Us Weekly broke the news that he was allegedly unfaithful in 2011, the twosome called it quits. The Ranch star married Mila Kunis in July 2015. They share two kids: Wyatt, 4, and Dimitri, 2. Kutcher, for his part, reflected on the divorce during an appearance on Dax Shepard's "Armchair Expert" podcast last year. "Right after I got divorced, I went to the mountains for a week by myself," Kutcher told Shepard in February 2018. "I did no food, no drink   just water and tea. I took all my computers away, my phone, my everything. I was there by myself, so there was no talking. I just had a notepad, a pen and water and tea   for a week." He referred to the trip as "really spiritual and kind of awesome." "I wrote down every single relationship that I had where I felt like there was some grudge or some anything, regret, anything," Kutcher explained. "And I wrote letters to every single person, and on day seven, I typed them all out and then sent them." While Moore certainly doesn't hold back in Inside Out, a source told Us earlier this month that the Kutcher isn't worried about the book. "Ashton knew what was coming. He had a heads up on what is in the book," the insider said on September 13. "He's not mad or disappointed. This is Demi's truth, and he always felt sympathetic toward her. He knows her story and that her upbringing was difficult." Inside Out is available now. Scroll through for 10 revelations from the book:
    # print(text)
    news['abstract_embeddings'] = news['abstract'].apply(get_embeddings)
    news['title_embeddings'] = news['title'].apply(get_embeddings)
    return news

def create_connection():
    """
    Creates a connection to the database by taking in a user password.
    """
    database_name = 'user_info'
    connection_user = 'admin'
    try:
        connection_password = input("Enter Password: ")
        connection_host = 'svc-57117697-8d22-448b-8b96-d3b2a46d0970-dml.aws-virginia-6.svc.singlestore.com'
        connection_port = '3306'
        connection_url = f"mysql+pymysql://{connection_user}:{connection_password}@{connection_host}:{connection_port}/{database_name}"
        db_connection = create_engine(connection_url)
        return db_connection
    except Exception as e:
        print(f"Exception {e} occured, try again.\n")
        return create_connection()

def push_to_DB(news : pd.DataFrame, behaviors : pd.DataFrame) -> None:
    """ 
    Pushes data to the SingleStore database for usage in modelling within docker containers and virtual machines.

    Args:
        news (pd.DataFrame) : The dataframe containing the data from the news csv.
        behaviors (pd.DataFrame) : The dataframe containing the data from the behaviors csv.

    Returns:
        None.
    """
    db_connection = create_connection()
    user_interaction_data = decompose_interactions(num_iterations=100000, news=news, behaviors=behaviors)
    user_interaction_data.to_sql('user_behaviors', if_exists='replace', con=db_connection, index=False)
    print("Push to Database Successful")

def create_interaction_counts():
    """
    Creates interaction counts dataframes for usage in timestamp analysis.
    """
    
    news = pd.read_csv('../MIND_small/csv/news.csv')
    copynews = news.set_index('news_id')
    behaviors = pd.read_csv('../MIND_small/csv/behaviors.csv')
    category_popularity = pd.read_csv('../MIND_small/csv/category_with_popularity.csv')
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

    # print(pd.unique(copynews['category']))
    behaviors[category_popularity.columns.to_list()] = behaviors.apply(lambda row : get_interaction_popularity(row, True), axis='columns', result_type='expand')
    behaviors.rename(columns={column : column + "_history" for column in category_popularity.columns.to_list()}, inplace=True)
    behaviors[category_popularity.columns.to_list()] = behaviors.apply(lambda row : get_interaction_popularity(row, False), axis='columns', result_type='expand')
    behaviors.rename(columns={column : column + "_impression" for column in category_popularity.columns.to_list()}, inplace=True)
    behaviors.to_csv("../MIND_small/csv/behaviors_with_individual_counts.csv")

def modify_hourly(behaviors):
    """
    Bins the time column of behaviors into hours so that insights on how time of day affects category popularity can be examined.
    
    Args:
        behaviors (pd.DataFrame) : Behaviors dataframe with impression specific category popularity counts.

    Returns:
        behaviors (pd.DataFrame) : Behaviors dataframe modified to include a binned time column. 
    """
    # Can be printed to determine the minimum and maximum timeframe for the data.
    times = [behaviors['time'].max(), behaviors['time'].min()]

    # going to want to set it so that time is only representing the date of the week and not hours, so cut_points can join the date with 00:00:00 for hourly predictions
    
    cut_points = pd.date_range(start='2019-11-09 00:00:00', end='2019-11-15 00:00:00', freq='h') # hourly ranges for the time of the behaviors dataset
    # going to want to adjust cutpoints so that we are specifically thinking of hours from 1 - 24 with 24 being midnight (0)
    
    # Create labels for the bins.
    bins_str = cut_points.astype(str).values
    labels = ['({}, {}]'.format(bins_str[i-1], bins_str[i]) for i in range(1, len(bins_str))]
    
    # Apply the bins to the time column.
    behaviors['hour'] = pd.cut(behaviors['time'], cut_points, labels=labels, include_lowest=True)
    
    # Modify the times so that it is only hours.
    behaviors['hour'] = behaviors['hour'].apply(lambda time_string : time_string.split(" ")[-1][:2])
    behaviors = behaviors.drop(columns=['Unnamed: 0','Unnamed: 0.1', 'impression_id', 'user_id', 'history', 'impressions', 'time'])
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