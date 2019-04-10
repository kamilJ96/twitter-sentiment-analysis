from flask import Flask, request, jsonify
from flask_cors import CORS
import nltk
import string
import re
import pickle

app = Flask(__name__)
CORS(app)

f = open('/home/kamikami/mysite/my_classifier.pickle', 'rb')
classifier = pickle.load(f)
f.close()

f = open('/home/kamikami/mysite/my_features.pickle', 'rb')
word_features = pickle.load(f)
f.close()

# Strip punctuation, remove words less than 3 characters long,
# and convert to lowercase
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

@app.route('/')
def hello_world():
    return 'Hello from Kamil!'

@app.route('/sentiment', methods=['GET', 'POST'])
def classify_sentiment():
    sentence = request.args.get('s')

    # Generate the list of stop words
    translator = str.maketrans('', '', string.punctuation)
    with open('/home/kamikami/mysite/stopwords.txt', 'r') as f:
        stop_words = [line.rstrip('\n').translate(translator) for line in f]

    # Process the string and score it based on the classifier
    processed_msg = process_sentence(sentence, stop_words)
    score = classifier.prob_classify(extract_features(processed_msg)).prob("positive")

    resp = {"value": score}
    return jsonify(resp)
