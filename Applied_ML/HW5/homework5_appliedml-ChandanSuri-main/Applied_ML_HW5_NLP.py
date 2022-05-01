#!/usr/bin/env python
# coding: utf-8

# # **Applied Machine Learning Homework 5**
# **Due 2 May,2022 (Monday) 11:59PM EST**
# 
# **Name: Chandan Suri, UNI: CS4090**

# ### Natural Language Processing
# We will train a supervised training model to predict if a tweet has a positive or negative sentiment.

# In[1]:


# Imports
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegressionCV

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

import string

random_seed = 42


# ####  **Dataset loading & dev/test splits**

# **1.1) Load the twitter dataset from NLTK library**

# In[2]:


import nltk
nltk.download('twitter_samples')
from nltk.corpus import twitter_samples 


# **1.2) Load the positive & negative tweets**

# In[3]:


all_positive_tweets = twitter_samples.strings('positive_tweets.json')
all_negative_tweets = twitter_samples.strings('negative_tweets.json')


# In[4]:


all_positive_tweets


# In[5]:


all_negative_tweets


# **1.3) Create a development & test split (80/20 ratio):**

# In[6]:


all_positive_tweets_df = pd.DataFrame(all_positive_tweets, columns = ["Tweets"])
all_positive_tweets_df["Sentiment_Label"] = "Positive"

all_negative_tweets_df = pd.DataFrame(all_negative_tweets, columns = ["Tweets"])
all_negative_tweets_df["Sentiment_Label"] = "Negative"


# In[7]:


print(f"Number of Positive Sentiment reviews are: {len(all_positive_tweets_df)}")
print(f"Number of Negative Sentiment reviews are: {len(all_negative_tweets_df)}")


# In[8]:


all_positive_tweets_df.head(10)


# In[9]:


all_negative_tweets_df.head(10)


# In[10]:


# Merging the dataframes
all_tweets_df = pd.concat([all_positive_tweets_df, all_negative_tweets_df])
all_tweets_df.reset_index = True


# In[11]:


# First we make all the tweets in lowercase
all_tweets_df.Tweets = all_tweets_df.Tweets.str.lower()


# In[12]:


all_tweets_df.head(10)


# In[13]:


dev_text, test_text, dev_y, test_y = train_test_split(all_tweets_df.Tweets, all_tweets_df.Sentiment_Label, 
                                                     test_size = 0.2, random_state = random_seed)


# In[14]:


print(f"The number of tweets in the development set are: {len(dev_text)}")
print(f"The number of tweets in the test set are: {len(test_text)}")
print(f"The number of labels in the development set are: {len(dev_y)}")
print(f"The number of labels in the test set are: {len(test_y)}")


# #### **Data preprocessing**
# We will do some data preprocessing before we tokenize the data. We will remove `#` symbol, hyperlinks, stop words & punctuations from the data. You can use the `re` package in python to find and replace these strings. 

# **1.4) Replace the `#` symbol with '' in every tweet**

# In[15]:


# For development
dev_text.replace('#', '', inplace = True, regex = True)

# For testing
test_text.replace('#', '', inplace = True, regex = True)


# **1.5) Replace hyperlinks with '' in every tweet**

# In[16]:


# For development
dev_text.replace('http[^ ]*', '', inplace = True, regex = True)

# For testing
test_text.replace('http[^ ]*', '', inplace = True, regex = True)


# **1.6) Remove all stop words**

# In[17]:


stop_words_joined = '|'.join(stopwords.words('english'))
stop_words_regex = f'\s({stop_words_joined})\s' 

# For development
dev_text.replace(stop_words_regex, ' ', inplace = True, regex = True)

# For testing
test_text.replace(stop_words_regex, ' ', inplace = True, regex = True)


# **1.7) Remove all punctuations**

# In[18]:


puncts_regex = f"[{string.punctuation}]"

# For development
dev_text.replace(puncts_regex, '', inplace = True, regex = True)

# For testing
test_text.replace(puncts_regex, '', inplace = True, regex = True)


# **1.8) Apply stemming on the development & test datasets using Porter algorithm**

# In[19]:


porter = PorterStemmer()

def stem_sentences(tweets):
    pre_processed_tweets = list()
    
    for tweet in tweets:
        tokenized_tweet = word_tokenize(tweet)
        stemmed_tweet = [porter.stem(token) for token in tokenized_tweet]
        pre_processed_tweets.append(" ".join(stemmed_tweet))
    
    return pre_processed_tweets


# In[20]:


dev_text = stem_sentences(dev_text)
test_text = stem_sentences(test_text)


# As we can see, all the data for tweets has been pre-processed completely!

# #### **Model training**

# **1.9) Create bag of words features for each tweet in the development dataset**

# In[26]:


count_vectorizer = CountVectorizer()
dev_X = count_vectorizer.fit_transform(dev_text)
test_cv_X = count_vectorizer.transform(test_text)


# **1.10) Train a supervised learning model of choice on the development dataset**

# In[27]:


lr_model_bow = LogisticRegressionCV(max_iter = 750, cv = 5, random_state = random_seed)
lr_model_bow = lr_model_bow.fit(dev_X, dev_y)


# **1.11) Create TF-IDF features for each tweet in the development dataset**

# In[28]:


tfidf_vectorizer = TfidfVectorizer()
dev_X = tfidf_vectorizer.fit_transform(dev_text)
test_tfidf_X = tfidf_vectorizer.transform(test_text)


# **1.12) Train the same supervised learning algorithm on the development dataset with TF-IDF features**

# In[29]:


lr_model_tfidf = LogisticRegressionCV(max_iter = 750, cv = 5, random_state = random_seed)
lr_model_tfidf = lr_model_tfidf.fit(dev_X, dev_y)


# **1.13) Compare the performance of the two models on the test dataset**

# In[30]:


print(f"The Accuracy Score for the Model (BOW) on Test Set is : {lr_model_bow.score(test_cv_X, test_y)*100}%")
print(f"The Accuracy Score for the Model (TFIDF) on Test Set is : {lr_model_tfidf.score(test_tfidf_X, test_y)*100}%")


# As seen above, the model is performing better with the TFIDF Vectorizer than the bag of words.
