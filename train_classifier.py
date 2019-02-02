"""
    Author: Kamil Jakrzewski

    This program takes an already-classified list of tweets (tweets that
    have been manually classified as positive or negative), and uses it
    to train the nltk naive-bayes classifier.
"""
import nltk
import json
import csv
import string
import re
import pickle

tweets = []
num_negative = 0
num_positive = 0


def process_tweet(tweet, stop_words):
    # Remove URLs
    tweet = re.sub(r"((www\.)|(?:\@|http?\:\/\/))\S+", "", tweet)
    
    # Remove Username mentions
    tweet = re.sub(r"@\S+", "", tweet)
    
    # Remove punctuation
    no_punc = "".join(char for char in tweet if char not in string.punctuation)

    # Make a list of all words in lowercase if they are longer than 2 letters
    words = []
    for word in no_punc.split():
        if len(word) > 2 and word not in stop_words and not re.search('^\d', word):
            # Replace 3 or more chars in a row with a single char
            word = re.sub(r"(.)\1{2,}", r"\1", word)
            words.append(word.lower())
    return words

stop = open('stop_words.pickle', 'rb')
stop_words = pickle.load(stop)

with open('fucking_big.csv', 'r', encoding='latin-1') as test_tweets:
    tweet_reader = csv.reader(test_tweets)

    for row in tweet_reader:
        if row[0] == '0':
            if num_negative > 50000:
                continue
            sentiment = 'negative'
            num_negative += 1
        elif row[0] == '4':
            if num_positive > 50000:
                continue
            sentiment = 'positive'
            num_positive += 1
        else:
            continue
        words = process_tweet(row[5], stop_words)
        tweets.append((words, sentiment))


def get_words_in_tweets(tweets):
    all_words = []
    for (words, sentiment) in tweets:
        all_words.extend(words)
    return all_words


def get_word_features(wordlist):
    wordlist = nltk.FreqDist(wordlist)
    word_features = wordlist.keys()
    return word_features


word_features = get_word_features(get_words_in_tweets(tweets))


def extract_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains(%s)' % word] = (word in document_words)
    return features

training_set = nltk.classify.apply_features(extract_features, tweets)
classifier = nltk.NaiveBayesClassifier.train(training_set)
f = open('my_classifier.pickle', 'wb')
pickle.dump(classifier, f)
f.close()

f = open('my_features.pickle', 'wb')
pickle.dump(list(word_features), f)
f.close()
