"""
    Author: Kamil Jakrzewski

    This program takes a JSON file of tweets, and strips all unnecessary
    metadata, and then pickles and saves it for later use.
"""
import pandas as pd
import pickle
import json

all_data = []
cols = ["created_at", "text"]
user_cols = ["followers_count", "statuses_count", "favourites_count", "verified", "time_zone"]

with open('twitter.json', 'r') as f:
    for line in f:
        data = json.loads(line)
        selected_data = []

        # Ignore tweets from London as most likely England-related
        if data["user"]["time_zone"] == "London" or "london" in data["text"].lower():
            continue

        # Append tweet and creation date
        for item in cols:
            selected_data.append(data[item].lower())

        # Append coordinates of tweeter if available
        if data["coordinates"]:
            selected_data.append(data["coordinates"]["coordinates"])
        else:
            selected_data.append(None)

        # Append tweeter statistics
        for item in user_cols:
            selected_data.append(data['user'][item])
        
        all_data.append(selected_data)


cols.append("coordinates")

all_data.append(list(cols + user_cols))

f = open('tweet_dump.pickle', 'wb')
pickle.dump(all_data, f)
f.close()

