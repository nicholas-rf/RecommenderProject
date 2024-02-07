# Whats new: Tuesday Feb 6th

## Modelling:
Added Tensorflow sequential recommender modelling docs that essentiall takes the user / query tower and makes it utilize a recurrent neural network for recommendations. It takes a sequence of user history and makes a recommendation from that 

## Data processing / EDA:
Worked heavily on making some charts for temporal data, specifically the popularity of specific categories given certain times.

## Potential:
Not much to report here, have considered implementing DLRM model with a tensorflow sequential stacked model which could be cool

# Whats new:

## Modelling:
Tensorflow model documentation for retrieval and ranking has been written so that we can write our own versions with the news dataset. Additionally I have started writing the docs for the sequential model which aims to predict the users next interaction item via RNN. There are more intermediate tensorflow tutorials on further feature extraction which I hope to get to sometime this week. 

## Data processing:
Not much on this front, the news dataset is in TSV format which doesn't contain columns, so I turned them into dataframes, added columns and exported as csv. 

## EDA:
I have started the EDA process by attempting to chart some intial distributions of categories and sub-categories in the news dataset. The current idea I had was to analyze the news and behavior datasets seperately at first, and then to analyze them within context of each other. Currently exploring data visualizations for attractive EDA. Worked on writing some more down for the report and thinking of some unique things we can create regarding interactions for the recommender system. Current thoughts include but are not limited to: a popularity column in the news dataset that is a count of how many users have interacted with it to a certain point, some way of recognizing the users preferred genres, a way to get the genres of user history, putting some sort of information into the news dataset that contains optimal times for certain categories. Also worth mentioning that we can utilize the sequencing of items in a history as a feature interaction. Could also attempt to bootstrap or concatenate some information onto the news dataset with user preferences and facts about them, ie their age, sexuality, etc. 


## Potential: 
After we have an established model pipeline going and are able to truly experiment, there is a column in the news dataset that contains the urls to specific articles. This could be utilized for some sort of web-scraping and a language model to get some key information.

##### also hi coby :3


##### interesting paper re creating good features within a recommender system with a neural network:
https://arxiv.org/abs/2008.00404


