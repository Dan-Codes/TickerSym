import re
from statistics import mean

import pandas as pd
import datetime
# pip install pandas-datareader
import pandas as pd
import pandas_datareader.data
from matplotlib import style
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
import miningTweets as mt
import oldTweets
import prettytable
import stock_market_lexicon
from textblob import TextBlob
import json

import csv

sid = SentimentIntensityAnalyzer()
words = stock_market_lexicon.additional
de_token = TreebankWordDetokenizer()
import time

style.use('ggplot')
ps = PorterStemmer()
client = mt.client
stopwords = set(stopwords.words("english"))


def filterWords(word_token):
    filtered_sentence = []
    for w in word_token:
        if w not in stopwords:
            # w = ps.stem(w)   # stemming api not very good ex. tragedy => tragedi
            filtered_sentence.append(w)

    return filtered_sentence


def getPolarity2(tweet):
    word_token = word_tokenize(tweet)  # tokenize sentence
    filtered = filterWords(word_token)  # filter the dumb words
    filtered = de_token.detokenize(filtered)  # un-tokenize
    # sentence = getFunction.cleanTweet(filtered)  # clean sentence
    ss = sid.polarity_scores(filtered)
    print(tweet, ss)
    return ss


def removeURL(tweet):
    result = re.sub(r"https:\S+", "", tweet)
    return result


def removeTicker(tweet):
    # ticker = "$"
    # result = re.sub(r"\S+", "", tweet)
    return tweet


def filterTweet(tweet):
    sentence = oldTweets.TweetAnalyzer().cleanTweet(tweet)  # clean sentence
    removedURL = removeURL(sentence)
    removedTicker = removeTicker(removedURL)
    word_token = word_tokenize(removedTicker)  # tokenize sentence
    filtered = filterWords(word_token)  # filter the dumb words
    filtered = de_token.detokenize(filtered)
    return removedTicker


def getPolarity(tweet):
    ss = sid.polarity_scores(tweet)
    if tweet == "":
        return ss
    compound = []
    tweet_token = word_tokenize(tweet)
    scores = [sid.polarity_scores(token) for token in tweet_token]
    compound = [score['compound'] for score in scores]
    m = mean(compound)
    ss['mean'] = m
    print(tweet, ss)
    return ss


def getDB(date):
    getFunction = oldTweets.TweetAnalyzer
    db = client['spx']
    mydb = db[date]
    total_polarity = 0
    neg_polarity = 0
    pos_polarity = 0
    count = 0

    for doc in mydb.find():
        ts = time.strftime('%Y-%m-%d %H:%M:%S', time.strptime(doc['created_at'],
                                                              '%a %b %d %H:%M:%S +0000 %Y'))  # get timestamp from database
        if ts.__contains__(date):
            tweet = doc['text']
            filtered = filterTweet(tweet)
            ss = getPolarity(filtered)
            # print(filtered, ss)
            if ss['compound'] != 0.0:
                total_polarity += ss['compound']
                neg_polarity += ss['neg']
                pos_polarity += ss['pos']
                count += 1

    total_polarity = total_polarity / count
    neg_pol = neg_polarity / count
    pos_pol = pos_polarity / count
    print("Polarity= ", total_polarity)
    polarity = dict()
    polarity['pos'] = str(pos_pol)
    polarity['neg'] = str(neg_pol)
    polarity['compound'] = str(total_polarity)
    return polarity


def fixDatabase(date):
    getFunction = oldTweets.TweetAnalyzer
    db = client['spx']
    mydb = db['tweets']
    total_polarity = 0
    count = 0
    for doc in mydb.find():
        # db.bios.find( { birth: { $gt: new Date('1940-01-01'), $lt: new Date('1960-01-01') } } )
        ts = time.strftime('%Y-%m-%d %H:%M:%S', time.strptime(doc['created_at'],
                                                              '%a %b %d %H:%M:%S +0000 %Y'))  # get timestamp from database
        if (ts.__contains__(date)):
            db[date].insert_one(doc)


def getPriceData():
    # sp500 = pandas_datareader.get_data_yahoo('%5EGSPC',
    #                                    start=datetime.datetime(2020, 4, 21),
    #                                    end=datetime.datetime(2020, 4, 25))   # api to get sp500 chart prices
    # print(sp500.head())

    df = pd.read_csv('sp500.csv', index_col='Date', parse_dates=True)

    for date in (df.index.array):
        date = (str(date)[0:10])
        polarity = getDB(date)
        print(date, polarity)

        # with open('sp500.csv', 'r') as f:
        # reader = csv.reader(f)
        # df.set_value(date, "compound_polarity", polarity)
        df.at[date, "compound_polarity"] = polarity['compound']
        df.at[date, "positive_polarity"] = polarity['pos']
        df.at[date, "negative_polarity"] = polarity['neg']

    df.to_csv('sp500.csv')


def tweets_to_dataframe(tweets):
    df = pd.DataFrame(data=[tweet.text for tweet in tweets], columns=['Tweets'])
    return df


def cleanTweet(tweet):
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())


def printDF(file_name):
    with open(file_name, "r") as fp:
        x = prettytable.from_csv(fp)

    print(x)


if __name__ == '__main__':
    sid.lexicon.update(words)
    getPriceData()
    printDF("sp500.csv")
