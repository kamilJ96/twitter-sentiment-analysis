# twitter-sentiment-analysis
Analyse whether a Tweet contains positive or negative sentiments based on the language used

First, the NLTK classifier must be trained so it can more accurately classify the sentiments of tweets. We take an already-classified file of tweets and feed it into the Naive-Bayes tool, helping it learn what makes a tweet either positive or negative. This is done in train_classifier.py.

We then take a JSON file of Tweets, these are processed and excess metadata is stripped from them in process_tweets.py.
This data is then pickled, and saved. analyse_tweets.py then unpickles all this data, and also takes in a command line argument that will be used to filter tweets based on that word. Only tweets that contain this word are included, and then NLTK (Natural Language Toolkit) is used to classify whether a given tweet is written positively or negatively. Finally, the distribution of ratings is plotted on a histogram and shown.

Created in my second year of University as a project for Elements of Data Processing.
