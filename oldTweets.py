import GetOldTweets3 as got # make sure to do pip install GetOldTweets3 first.
import json
import datetime
import re
import time

# make sure to pip install these guys
import numpy as np
import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize  ## make sure all stuff is downloaded
from textblob import TextBlob

from pymongo import MongoClient


pd.set_option("display.max_columns", 100)
pd.set_option("display.max_colwidth", 50)

#print("Current Time: ", datetime.datetime.now())
client = MongoClient("mongodb+srv://dbSPX:SQSeKptrpjt6Bi7F@cluster0-p4uhp.mongodb.net/test?retryWrites=true&w=majority")
db = client.test
#print(db)

######################

class TweetAnalyzer():

    # placing raw tweet data into a data_frame
    def tweetsToDataFrame(self, tweets):
        df = pd.DataFrame(data=[ tweets[tweet]["text"] for tweet in tweets], columns=['tweets'])
        ##df['id']        = np.array([tweets[tweet]["id"] for tweet in tweets])
        ##df['date']      = np.array([tweets[tweet]["date"] for tweet in tweets])
        ##df['favorites'] = np.array([tweets[tweet]["favorites"] for tweet in tweets])
        ##df['retweets']  = np.array([tweets[tweet]["retweets"] for tweet in tweets])
        ##df['hashtags']  = np.array([tweets[tweet]["hashtags"] for tweet in tweets])

        return df

    # gets rid of non-alphabetical words
    def cleanTweet(self, tweet):
        return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())

    #using Textblob to analyze sentiment
    def analyze_sentiment(self, tweet):
        analysis = TextBlob(self.cleanTweet(tweet))

        if analysis.sentiment.polarity > 0:     return 1
        elif analysis.sentiment.polarity == 0:  return 0
        else:                                   return -1

############ CONSTANTS

# iP = inflectionPoints, month range
iP = {"Great Recession 2008"  : ("2007-12-01", "2007-12-05"),
      "Coronavirus"           : ("2020-02-20", "2020-02-21"), #("2020-01-01", "2020-04-14")
      "China-US Trade War"    : ("2018-12-01", "2018-12-31")
     }

#
#2/21 - 3/22 cv
#dateRange = [date.strftime('%Y-%m-%d') for date in pd.date_range(start="2020-03-22", end="2020-03-23")] ## upperbound not included

#10/15 - 11/15 flat
dateRange = [date.strftime('%Y-%m-%d') for date in pd.date_range(start="2019-10-15", end="2019-11-16")] ## upperbound not included


hashtagList = ["S&P500", "SP500", "$SPX", "$SPY"]
keywords = ["bullish", "bearish", "support line", "resistance line"]


if __name__ == '__main__':

    date_i = 0
    while date_i < len(dateRange)-1:
        print("Date: " + dateRange[date_i])
        
        #event = "Coronavirus"
        fromDate = dateRange[date_i] #iP[event][0]
        toDate = dateRange[date_i + 1] #iP[event][1]
        query = " OR ".join(hashtagList) # see Twitter documentation for different search operators
        maxTweets = 3000 # comment this out if unlimited

        tweetCriteria = got.manager.TweetCriteria().setQuerySearch(query)\
                                                   .setSince(fromDate)\
                                                   .setUntil(toDate)\
                                                   .setMaxTweets(maxTweets) # comment this if unnecessary

        ###### RESULTS
        masterTweetsList = []
        print("Getting " + str(maxTweets) + " historical tweets about from: " + fromDate + " to: " + toDate)

        mydb = client["spx"]
        count = 0
        for tweet in got.manager.TweetManager.getTweets(tweetCriteria):
            if count == 0: print("Inserting into db...")

            # Tweet properties (see documentation for full list of properties: https://pypi.org/project/GetOldTweets3/)
            currentTweet = {
                "historical": True,
                "date": (tweet.date).strftime('%Y-%m-%d'),
                "username": tweet.username,
                "text": tweet.text,
                
                "id": tweet.id,
                "permalink": tweet.permalink,
                "to": tweet.to,
                "retweets": tweet.retweets,
                "favorites": tweet.favorites,
                "mentions": tweet.mentions,
                "hashtags": tweet.hashtags,
                "geo": tweet.geo }

            masterTweetsList.append(currentTweet)
            count += 1

        mycol = mydb[dateRange[date_i]]
        mycol.insert_many(masterTweetsList)

        print("Current Time: ", datetime.datetime.now())
        print("finished mongodb upload")

        date_i += 1

        if date_i > 0 and date_i % 4 == 0:
            print("Sleeping in case of rate limit...")
            print("Current Time: ", datetime.datetime.now())
            time.sleep(900)


