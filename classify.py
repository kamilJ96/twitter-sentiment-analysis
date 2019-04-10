import nltk
import pickle
import string
import re

f = open('my_classifier.pickle', 'rb')
classifier = pickle.load(f)
f.close()

f = open('my_features.pickle', 'rb')
word_features = pickle.load(f)
f.close()

def process_sentence(sentence, stop_words):
    no_punc = "".join(char for char in sentence if char not in string.punctuation)
    words = []
    for word in no_punc.split():
        if len(word) > 2 and word not in stop_words:
            word = re.sub(r"(.)\1{2,}", r"\1", word)
            words.append(word.lower())

    return words

def extract_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains(%s)' % word] = (word in document_words)
    return features

translator = str.maketrans('', '', string.punctuation)
with open('stopwords.txt', 'r') as f:
    stop_words = [line.rstrip('\n').translate(translator) for line in f]
    
processed_msg = process_sentence(message, stop_words)
score = classifier.prob_classify(extract_features(processed_msg)).prob("positive")
print("That sentiment is {}% positive.\n".format(score * 100))
