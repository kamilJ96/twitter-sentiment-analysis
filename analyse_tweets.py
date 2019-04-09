"""
    Author: Kamil Jakrzewski (760596)

    This program takes a list of tweets supplied with a pickle, and a
    given word supplied through the command line, and using NLTK analyses each 
    tweet that contains the given word and rates their sentiment on a scale 
    of 0 to 1, 0 being most negative and 1 being most positive, finally 
    plotting the distribution on a histogram.
"""
import sys
import nltk
import numpy as np
import matplotlib.pyplot as plt
import re
import string
import pickle
import pandas as pd

# Parse a tweet into an object that is able to be analysed using
# natural language tools
def process_tweet(tweet, stop_words):
    # Remove URLs
    tweet = re.sub(r"((www\.)|(http(s)??\:\/\/))\S+", "", tweet)

    # Remove Username mentions
    tweet = re.sub(r"@\S+", "", tweet)
    tweet = re.sub(r"[^a-zA-Z ]", "", tweet)

    # Remove punctuation
    no_punc = "".join(char for char in tweet if char not in string.punctuation)

    # Make a list of all words in lowercase if they are longer than 2 letters
    words = []
    for word in no_punc.split():
        if len(word) > 2 and word not in stop_words and not re.search('^\d', word):
            # Replace 3 or more chars in a row with a single word string
            word = re.sub(r"(.)\1{2,}", r"\1", word)
            words.append(word.lower())

    return words

def extract_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains(%s)' % word] = (word in document_words)
    return features

f = open('my_classifier.pickle', 'rb')
classifier = pickle.load(f)
f.close()

f = open('my_features.pickle', 'rb')
word_features = pickle.load(f)
f.close()

# Stop words are short, functional words that are useless to the analysis
# of natural language, e.g. 'the', 'a', 'for', etc.; these are removed from
# tweets for efficacy.
f = open('final_stop.pickle', 'rb')
stop_words = set(pickle.load(f))
f.close()

f = open('tweet_dump.pickle', 'rb')
data = pickle.load(f)
f.close()

cols = data[-1]
data = data[:-1]

tweets = pd.DataFrame(data, columns = cols)
tweets = tweets.drop_duplicates("text")

# The below code analyses the sentiment expressed towards tweets relating
# to a given word supplied through the command line as an argument

muse_tweets = tweets[tweets["text"].str.contains(str(sys.argv[1]))]

scores = []
pos_tweets = []
neg_tweets = []

for row in muse_tweets.itertuples():
    processed_tweet = process_tweet(row[2], stop_words)
    score = classifier.prob_classify(extract_features(processed_tweet)).prob("positive")
   
    if score > 0.5:
        pos_tweets.append(row[2])
    else:
        neg_tweets.append(row[2])

    scores.append(score)

print('positive = {}\nnegative = {}'.format(len(pos_tweets), len(neg_tweets)))
print(pos_tweets[:10])

plt.hist(scores, bins=10, normed=True)
plt.xlabel('Positivity score')
plt.ylabel('Probability')
plt.show()
