# Import all dependencies
from pandas import Series

import pandas as pd

# For text cleaning
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

import nltk
import string
import re

# For text analysis
from nltk.collocations import *
from nltk.text import Text 

import itertools
import collections

# For text classification
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, silhouette_score

import numpy as np

# For sentimental analysis
from gensim.models.phrases import Phrases, Phraser
from collections import defaultdict  # For word frequency
from gensim.models import Word2Vec
from time import time  # To time our operations

import multiprocessing

# For visualization
from matplotlib.ticker import MaxNLocator
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import matplotlib.pyplot as plt

# Text cleaning
stop_words = set(stopwords.words('english'))

def remove_(txt):
    "remove hashtag, @user, link of a post using regular expression"
    return ' '.join(
        re.sub(
            "(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ", txt
        ).split())

def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)

def text_cleaner(text):
    # split into words
    tokens = word_tokenize(text)
    # convert to lower case
    tokens = [w.lower() for w in tokens]
    # remove punctuation from each word
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in tokens]
    # remove remaining tokens that are not alphabetic
    words = [word for word in stripped if word.isalpha()]
    # filter out stop words
    words = [w for w in words if not w in stop_words]
    # filter out word searched, e.g. juice and beverage
    addntl_stopwords = ['juices', 'beverages', 'juice', 'beverage']
    words = [w for w in words if not w in addntl_stopwords]
    # lemmatize the words
    lemmatizer = WordNetLemmatizer()
    lemmatized = [lemmatizer.lemmatize(word, get_wordnet_pos(word)) for word in words]
    #filter out one-letter words
    words = [w for w in lemmatized if len(w) > 1]
    return ' '.join(words)

def dataframe_cleaner(dataframe):
    dataframe['cleaned_content'] = dataframe['Content'].apply(remove_)
    dataframe['cleaned_content'] = dataframe['cleaned_content'].apply(text_cleaner)

    list_duplicates = [1 if x else 0 for x in dataframe.duplicated(subset='cleaned_content', keep='first')]
    dataframe['duplicates'] = list_duplicates
    dataframe.to_csv('data/cleaned.csv', index=False)
    return dataframe

def word_frequency(dataframe):
    # Create a list of lists containing words per post
    words_per_post = [word.split() for word in dataframe['cleaned_content']]

    # List of all words across posts
    all_words = list(itertools.chain(*words_per_post))

    # Create counter
    counts_ = collections.Counter(all_words)

    df_clean = pd.DataFrame(counts_.most_common(), columns=['words', 'count'])
    df_clean.head()
    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot horizontal bar graph
    df_clean.sort_values(by='count', ascending=False).head(15).sort_values(
        by='count').plot.barh(x='words', y='count', ax=ax, color="purple")

    ax.set_title("Top 15 Words Found in Posts")
    plt.savefig('img/word_frequency.png')
    plt.show();

def pairwise(t):
    x = t[:-1]
    y = t[1:]
    return zip(x, y)

def triwise(t):
    x = t[:-2]
    y = t[1:-1]
    z = t[2:]
    return zip(x, y, z)

def collocate_df(dataframe):
    df_bigrams = pd.concat(
    [Series(
        row['Post ID'], pairwise(row['cleaned_content'].split(' '))
    ) for _, row in dataframe.iterrows()
    ]
    ).reset_index()
    df_bigrams['bigrams'] = df_bigrams['index'].apply(lambda x: remove_(str(x)))
    df_bigrams.drop('index', axis=1, inplace=True)
    
    df_trigrams = pd.concat(
    [Series(
        row['Post ID'], triwise(row['cleaned_content'].split(' '))
    ) for _, row in dataframe.iterrows()
    ]
    ).reset_index()
    df_trigrams['trigrams'] = df_trigrams['index'].apply(lambda x: remove_(str(x)))
    df_trigrams.drop('index', axis=1, inplace=True)
    
    df_bigrams.to_csv('data/bigrams.csv', index=False)
    df_trigrams.to_csv('data/trigrams.csv', index=False)
    return df_bigrams, df_trigrams

def vectorize(words):
    """Returns a vectorized bag of words and the list of nonzero features"""
    tfidf_vectorizer = TfidfVectorizer(stop_words='english',
                                   token_pattern=r'[a-z-]+')
    bow = tfidf_vectorizer.fit_transform(words)
    nonzeros = bow.sum(axis=1).nonzero()[0]
    bow = bow[nonzeros]
    return bow, nonzeros

def cluster_range(X, clusterer, k_start, k_stop):
    """Get the range of clusters per internal validation"""
    scs = []
    inertias = []
    
    for i in range(k_start, k_stop+1):

        new_clusterer = clusterer
        new_clusterer.n_clusters = i
        
        y = new_clusterer.fit_predict(X)

        scs.append(silhouette_score(X, y))     
        inertias.append(new_clusterer.inertia_)
        
    ret = {'scs':scs,
     'inertias':inertias
    }
    return ret

def plot_internal(inertias, scs):
    """Plot internal validation values"""
    fig, ax = plt.subplots()
    ks = np.arange(2, len(inertias)+2)
    ax.plot(ks, inertias, '-o', label='SSE')
    ax.set_xlabel('$k$')
    ax.set_ylabel('SSE')
    lines, labels = ax.get_legend_handles_labels()
    ax2 = ax.twinx()
    ax2.plot(ks, scs, '-ko', label='Silhouette coefficient')
    ax2.set_ylabel('Silhouette')
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines+lines2, labels+labels2)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    return ax

def df_clustered(df, nonzeros, clusters):
    """Returns a DataFrame of titles with cluster number"""
    clustered_table = df.iloc[list(nonzeros)]
    clustered_table['Cluster'] = clusters
    clustered_table = clustered_table.reset_index(drop=True)
    
    name_clusters = []
    for i in range(clustered_table.shape[0]):
        if clustered_table['Cluster'].loc[i] == 0:
            name_clusters.append('Fruit Juice')
        elif clustered_table['Cluster'].loc[i] == 1:
            name_clusters.append('Machines Involving Juice')
        elif clustered_table['Cluster'].loc[i] == 2:
            name_clusters.append('Drinking Juice')
        elif clustered_table['Cluster'].loc[i] == 3:
            name_clusters.append('Making Juice')
        elif clustered_table['Cluster'].loc[i] == 4:
            name_clusters.append('GAGT Users')
        else:
            name_clusters.append('Orange Juice')
    clustered_table['cluster_name'] = name_clusters
    
    clustered_table.to_csv('data/clustered.csv')
    return clustered_table

def visualize(df, n):
    """Returns a wordcloud of titles for visualization"""
    clean_string_group = re.sub('\W', ' ', ' '.join(
        df.groupby('Cluster').get_group(n).cleaned_content))
    wordcloud = WordCloud(max_words=300, background_color='white', scale=3,
                          colormap='viridis').generate(clean_string_group)
    fig = plt.figure()
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()
    
def create_tfidf_dictionary(x, transformed_file, features):
    '''
    create dictionary for each input sentence x, where each word has assigned its tfidf score
    
    inspired  by function from this wonderful article: 
    https://medium.com/analytics-vidhya/automated-keyword-extraction-from-articles-using-nlp-bfd864f41b34
    
    x - row of dataframe, containing sentences, and their indexes,
    transformed_file - all sentences transformed with TfidfVectorizer
    features - names of all words in corpus used in TfidfVectorizer
    '''
    vector_coo = transformed_file[x.name].tocoo()
    vector_coo.col = features.iloc[vector_coo.col].values
    dict_from_coo = dict(zip(vector_coo.col, vector_coo.data))
    return dict_from_coo

def replace_tfidf_words(x, transformed_file, features):
    '''
    replacing each word with it's calculated tfidf dictionary with scores of each word
    x - row of dataframe, containing sentences, and their indexes,
    transformed_file - all sentences transformed with TfidfVectorizer
    features - names of all words in corpus used in TfidfVectorizer
    '''
    dictionary = create_tfidf_dictionary(x, transformed_file, features)   
    return list(map(lambda y:dictionary[f'{y}'], x.cleaned_content.split()))

def replace_sentiment_words(word, sentiment_dict):
    '''
    replacing each word with its associated sentiment score from sentiment dict
    '''
    try:
        out = sentiment_dict[word]
    except KeyError:
        out = 0
    return out

def get_sentiments(df):
    sent = [row.split() for row in df['cleaned_content']]
    phrases = Phrases(sent, min_count=30, progress_per=10000)
    bigram = Phraser(phrases)
    sentences = bigram[sent]
    word_freq = defaultdict(int)
    for sent in sentences:
        for i in sent:
            word_freq[i] += 1
    cores = multiprocessing.cpu_count() # Count the number of cores in a computer
    w2v_model = Word2Vec(min_count=20,
                     window=2,
                     size=300,
                     sample=6e-5, 
                     alpha=0.03, 
                     min_alpha=0.0007, 
                     negative=20,
                     workers=cores-1)
    t = time()

    w2v_model.build_vocab(sentences, progress_per=10000)

    print('Time to build vocab: {} mins'.format(round((time() - t) / 60, 2)))
    t = time()

    w2v_model.train(sentences, total_examples=w2v_model.corpus_count, epochs=30, report_delay=1)

    print('Time to train the model: {} mins'.format(round((time() - t) / 60, 2)))
    w2v_model.init_sims(replace=True)
    model = KMeans(n_clusters=2, max_iter=1000, random_state=True, n_init=50).fit(X=w2v_model.wv.vectors)
    positive_cluster_center = model.cluster_centers_[0]
    negative_cluster_center = model.cluster_centers_[1]
    words = pd.DataFrame(w2v_model.wv.vocab.keys())
    words.columns = ['words']
    words['vectors'] = words.words.apply(lambda x: w2v_model.wv[f'{x}'])
    words['cluster'] = words.vectors.apply(lambda x: model.predict(np.array(x).reshape(1, -1)))
    words.cluster = words.cluster.apply(lambda x: x[0])
    words['cluster_value'] = [1 if i==0 else -1 for i in words.cluster]
    words['closeness_score'] = words.apply(lambda x: 1/(model.transform([x.vectors]).min()), axis=1)
    words['sentiment_coeff'] = words.closeness_score * words.cluster_value
    tfidf = TfidfVectorizer(tokenizer=lambda y: y.split(), norm=None)
    tfidf.fit(df['cleaned_content'])
    features = pd.Series(tfidf.get_feature_names())
    transformed = tfidf.transform(df['cleaned_content'])
    replaced_tfidf_scores = df.apply(lambda x: replace_tfidf_words(x, transformed, features), axis=1)
    sentiment_dict = dict(zip(words.words.values, words.sentiment_coeff.values))
    replaced_closeness_scores = df['cleaned_content'].apply(
        lambda x: list(map(lambda y: replace_sentiment_words(y, sentiment_dict), x.split())))
    replacement_df = pd.DataFrame(
    data=[replaced_closeness_scores, replaced_tfidf_scores, df['cleaned_content']]).T
    replacement_df.columns = ['sentiment_coeff', 'tfidf_scores', 'sentence']
    replacement_df['sentiment_rate'] = replacement_df.apply(lambda x: np.array(x.loc['sentiment_coeff']) @ np.array(x.loc['tfidf_scores']), axis=1)
    replacement_df['sentimet'] = replacement_df['sentiment_rate'].apply(lambda x: "positive" if x > 0 else "negative" )
    replacement_df.to_csv('data/sentiments.csv', index=False)
    return replacement_df

# Load the dataset
df = pd.read_excel('data/Social_Dataset.xlsx', sheet_name='Raw Data')

df = dataframe_cleaner(df)
df = df[df['duplicates']==0].reset_index(drop=True)

word_frequency(df)

df_bigrams, df_trigrams = collocate_df(df)

# Vectorize the cleaned data
# Get only the nonzeros features in the vectorized bag of words
bow_X, nonzeros = vectorize(df.cleaned_content)

# Find clusters using internal validation
# Sum of squares distances to centroids
# Silhouette coefficient
res_posts = cluster_range(bow_X, KMeans(random_state=1337), 2, 11)
plot_internal(res_posts['inertias'], res_posts['scs'])

# Choose five as number of clusters
kmeans_X = KMeans(random_state=1337, n_clusters=6)
y_predict_X = kmeans_X.fit_predict(bow_X)

# create a new DataFrame containing titles with cluster number (from 0 to 5)
df_new = df_clustered(df, nonzeros, y_predict_X)

df_sentiments = get_sentiments(df)

print('Please see my notebook for better documentation :D')