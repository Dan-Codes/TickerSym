import GetOldTweets3 as got # make sure to do pip install GetOldTweets3 first.
import json
import datetime
import re

# make sure to pip install these guys
import numpy as np
import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize  ## make sure all stuff is downloaded
from textblob import TextBlob


pd.set_option("display.max_columns", 100)
pd.set_option("display.max_colwidth", 50)

######################

class TweetAnalyzer():
    def tweetsToDataFrame(self, tweets):
        df = pd.DataFrame(data=[ tweets[tweet]["text"] for tweet in tweets], columns=['tweets'])
        ##df['id']        = np.array([tweets[tweet]["id"] for tweet in tweets])
        ##df['date']      = np.array([tweets[tweet]["date"] for tweet in tweets])
        ##df['favorites'] = np.array([tweets[tweet]["favorites"] for tweet in tweets])
        ##df['retweets']  = np.array([tweets[tweet]["retweets"] for tweet in tweets])
        ##df['hashtags']  = np.array([tweets[tweet]["hashtags"] for tweet in tweets])

        return df

    def cleanTweet(self, tweet):
        return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())

    def analyze_sentiment(self, tweet):
        analysis = TextBlob(self.cleanTweet(tweet))

        if analysis.sentiment.polarity > 0:     return 1
        elif analysis.sentiment.polarity == 0:  return 0
        else:                                   return -1

############ CONSTANTS

# iP = inflectionPoints, month range
iP = {"Great Recession 2008"  : ("2007-12-01", "2007-12-05"),
      "Coronavirus"           : ("2020-03-23", "2020-03-24"), #("2020-01-01", "2020-04-14")
      "China-US Trade War"    : ("2018-12-01", "2018-12-31")
     }

hashtagList = ["S&P500", "SP500", "$SPX", "$SPY"]
keywords = ["bullish", "bearish", "support line", "resistance line"]


if __name__ == '__main__':
    
    event = "Coronavirus"
    fromDate = iP[event][0]
    toDate = iP[event][1]
    query = " OR ".join(hashtagList) # see Twitter documentation for different search operators
    maxTweets = 1000 # comment this out if unlimited

    tweetCriteria = got.manager.TweetCriteria().setQuerySearch(query)\
                                               .setSince(fromDate)\
                                               .setUntil(toDate)\
                                               .setMaxTweets(maxTweets) # comment this if unnecessary

    ###### RESULTS
    masterTweets = {}
    print("Getting " + str(maxTweets) + " historical tweets about: " + event + " from: " + fromDate + " to: " + toDate)

    count = 0
    for tweet in got.manager.TweetManager.getTweets(tweetCriteria):
        #print("Tweet " + str(count))
        print(tweet)

        # Tweet properties (see documentation for full list of properties: https://pypi.org/project/GetOldTweets3/)
        masterTweets[tweet.id] = {
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

        count += 1


    tweet_analyzer = TweetAnalyzer()
    df = tweet_analyzer.tweetsToDataFrame(masterTweets) ## data frame
    df['sentiment']  = np.array([tweet_analyzer.analyze_sentiment(tweet) for tweet in df["tweets"]])

    print(df.head(10))
    print("Sentiment average: " + str(np.mean(df["sentiment"])))
    
    #f = open("historical_tweets.txt", "w")
    #json.dump(masterTweets, f)
    #f.close()
