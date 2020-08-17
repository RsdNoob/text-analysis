# Unsupervised Analysis on Public Posts about Juice using K-Means Clustering

## Executive Summary

Most of the time, text data doesn't have a label. There are basic analysis that can be done in this kind of data like word frequency and collocation. However, advance analysis can also be done if the data is labelled. These kind of analysis includes text classification and sentiment analysis. Hence, this project uses an unsupervised learning method to label text data. The data is from public posts in Singapore around juices and beverages. The timeframe of the data is from November 2018 to May 2019. Basic analysis will be done first before doing the advanced analysis. Using K-means clustering to classify, it showed five categories of posts which were **fruit juices**, **shared ideas about juices**, **drinking juices**, **making juices**, **GAGT Users** and **Orange Juices**. After doing this, another K-means clustering is done to define the sentiments per post. It showed that discussion of juice in Singapore were more negative in sentiment. This is mostly due to dirty orange juice given out by vending machines. Further improvements can be done like using other clustering methods to get better classification. Also, a dashboard of this project is published at <a href="https://public.tableau.com/profile/rsdnoob#!/vizhome/SingaporeanPostsaboutJuiceSentimentalAnalysis/Dashboard1  target="_blank">my Tableau dashboard</a>.

## Dataset

The dataset contains public posts from the top social media channels in Singapore around juices & beverages. It contains 13 columns which are:

- Post ID: identifier for each row in the dataset
- Post Date: date on which the social media post was made
- Title: title of the social media thread
- Content: content of the social media thread
- Post Type: indicates whethere the social media post is a original/first content of a thread (ARTICLE) or wether the post is a comment to the original/first content of a thread
- URL: the link to the post
- Channel Name: the name of the social media page where the post was made
- Channel Country: the country of origin of the social media post
- Channel Site Type: the social media  site where the post was made
    - SOCIAL_NETWORKING = Facebook
    - MICROBLOG = Twitter
    - FORUM = Forums
- Channel Language: The language of the post
- Channel URL: The link/url of the social media page where the post was made
- Voice Name: the name of the social media profile which made the post

Most of the data processing will be done in **Content** column. Due to limited time, there is little data exploration on the other columns.

**pandas** python module will be used to open and manipulate the data.

The dataset contains **8072** posts. However, this will be trim down later after text cleaning.

Most of the language used in this dataset is English. This information will be used later for text cleaning.

There are more comments than articles in this dataset. It can be useful for removing contents that are duplicate of another content from different users.

## Text cleaning

Cleaning the data is important since it can reduce the insights that can be discovered from it. Below are the steps to clean the text data:

- remove hashtag, @user, link of a post: this will dilute the word frequency since there can be some URLs that are shared in multiple posts
- convert text to lower case: this will be used since capitalized and non-capitalized words are still the same word after getting the lower case
- remove punctuations attached to each word: remove dots or commas
- remove remaining characters that are not alphabetic: remove special characters like emojis transformed into some special character
- filter out stopwords: as mentioned earlier, English is the most used language in this dataset. Hence, English stopwords will be used here
- filter out searched words: The words **juice**, **juices**, **beverage** and **beverages** will be removed since it will affect the word count. 
- lemmatize the words: this method is used to get the root of a word
- filter out one-letter words: remove one-letter word generated after lemmatazing the words

After this, basic and advance analysis can now be done.

## Basic Analysis

Word frequency and concordance will be done in this part.

### Word frequency

This analysis will be useful to know which words are usually associated to a post about juices.

### Collocation

Collocation is useful if the word frequency doesn't make sense. It is done by getting pairs (bi-gram) or triwise (tri-gram) of words in a single sentence or post.

## Advanced Analysis (Unsupervised)

Unsupervised means there are no labels in the dataset and it will be created using data science techniques. The method that will be used in this project is K-means clustering. There other clustering methods, however, due to time constraint of this project, this is the only clustering method that will  be used here.

### Text Classification

The first step in this analysis is to get trainable data. This means to transform the text data into numerical values.

After transforming the data, the next step is to identify the number of clusters. As of now, there are no automatic methods to do this. However, there are statistical tests that can be done to make identifying the number of clusters easier. In this projet, calinski harabasz ascore and silhouette score will be used.

*Use this as a reference:*
- https://scikit-learn.org/stable/modules/generated/sklearn.metrics.calinski_harabasz_score.html
- https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html

After identifying the number of clusters, it is now possible to train the K-means clustering.

### Sentiment Analysis

After text classification, sentiment analysis can also be done. Sentiment analysis is usually done with labelled data. However, in this project, NLTK and K-means clustering will be combined, so that it will be possible to generate labelled data. Most of the insights in this project can be seen at <a href="https://public.tableau.com/profile/rsdnoob#!/vizhome/SingaporeanPostsaboutJuiceSentimentalAnalysis/Dashboard1" target="_blank">my Tableau dashboard</a>.

*Note: Check the **text_analysis.py** to see the codes used in this analysis.*

The pictures below are from the dashboard created using this data.

<img src="img/pic1.PNG" />

<img src="img/pic2.PNG" />
          
<img src="img/pic3.PNG" />

Here is the link on the post on February 16, 2019: https://web.facebook.com/permalink.php?story_fbid=2880190185455025&id=1993145654159487&_rdc=1&_rdr.

## Insights

- Doing an exploratory data analysis (EDA) on the data is useful since it can later in improving the data, e.g. text cleaning.
- Text cleaning is important in text analysis since it improves or worsens the discovery of insgihts.
- Even though it is **Basic** Analysis, there are still a lot of insights that can be discovered.
- If the words **juice**, **juices**, **beverage** and **beverages** are not removed, they will show up as the most frequent words.
- Removing duplicates is important in text cleaning, since if it is not done, then insights from collocations will be affected. It is possible that useless bi-grams or tri-grams may show up.
- Unsupervised learning is useful if there is no labelled data available.
- K-means clustering is helpful for fast implementation of unsupervised learning. However, there are also cons in using it.
- The best take away part in this project is the sentiment analysis. It discovered the Social Media post from Frebruary 16, 2019. **Hence, it proves that the sentiment analysis in this project is useful.**

## Recommendations

- Use other clustering methods for unsupervised learning.
- Use Python's **Dash** to visualize, since it offers more control to the user.

## References

- https://gitlab.datascience.aim.edu/jcuballes/finding-subreddit-topics-through-kmeans-clustering
- https://towardsdatascience.com/unsupervised-sentiment-analysis-a38bf1906483
- https://www.flerlagetwins.com/2019/09/text-analysis.html
- https://monkeylearn.com/text-analysis/
