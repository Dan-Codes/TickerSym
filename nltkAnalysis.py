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

import matplotlib.pyplot as plt
import numpy as np

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


# dont worry about this
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
    result = re.sub(r"http:\S+", "", tweet)
    return result


def removeTicker(tweet):
    newArray = []
    i = 0
    while i < len(tweet):
        if tweet[i] != "$" and tweet[i] != "#":
            newArray.append(tweet[i])
            i += 1
        else:
            i += 2

    return newArray


def filterTweet(tweet):
    removedURL = removeURL(tweet) # Removes url from the tweet
    word_token = word_tokenize(removedURL)  # tokenize sentence
    #filtered = filterWords(word_token)  # filter the dumb words
    removedTicker = removeTicker(word_token)
    filtered = de_token.detokenize(removedTicker)
    sentence = oldTweets.TweetAnalyzer().cleanTweet(filtered)  # clean sentence
    return sentence


def getPolarity(tweet):
    ss = sid.polarity_scores(tweet)
    analysis = TextBlob(tweet)
    if tweet == "":
        return ss
    compound = []
    tweet_token = word_tokenize(tweet)
    scores = [sid.polarity_scores(token) for token in tweet_token]
    compound = [score['compound'] for score in scores]
    m = mean(compound)
    ss['mean'] = m
    ss['textblob_polarity'] = analysis.sentiment.polarity
    ## print(tweet, ss)
    return ss


def getDB(date):
    getFunction = oldTweets.TweetAnalyzer
    db = client['spx']
    mydb = db[date]
    textBlob_pol = []
    total_polarity = 0
    neg_polarity = 0
    pos_polarity = 0
    count = 0

    for doc in mydb.find():
        if not doc["historical"]:
            ts = time.strftime('%Y-%m-%d %H:%M:%S', time.strptime(doc['created_at'],
                                                              '%a %b %d %H:%M:%S +0000 %Y'))  # get timestamp from database
        else: ts = doc["date"]
        
        if ts.__contains__(date):
            tweet = doc['text']
            filtered = filterTweet(tweet)
            ss = getPolarity(filtered)
            # print(filtered, ss)
            if ss['compound'] != 0.0:
                total_polarity += ss['compound']
                neg_polarity += ss['neg']
                pos_polarity += ss['pos']
                textBlob_pol.append(ss['textblob_polarity'])
                count += 1


        mydb.update_one({"id": doc["id"]}, { '$push' : {'polarity_list' : ss['compound']}}, upsert=True)
        #mydb.update(document=doc, { '$push' : {'polarity_list' : ss['compound']}}, upsert=True)

    total_polarity = total_polarity / count
    neg_pol = neg_polarity / count
    pos_pol = pos_polarity / count
    print("Polarity= ", total_polarity)
    polarity = dict()
    polarity['pos'] = str(pos_pol)
    polarity['neg'] = str(neg_pol)
    polarity['compound'] = str(total_polarity)
    polarity['textBlob_pol'] = str(mean(textBlob_pol))
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


def getPriceData(f):
    # sp500 = pandas_datareader.get_data_yahoo('%5EGSPC',
    #                                    start=datetime.datetime(2020, 4, 21),
    #                                    end=datetime.datetime(2020, 4, 25))   # api to get sp500 chart prices
    # print(sp500.head())

    df = pd.read_csv(f, index_col='Date', parse_dates=True) #'sp500.csv'

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
        df.at[date, "textBlob_polarity"] = polarity['textBlob_pol']

    df.to_csv(f)

def make_dataframe_from_stock_tweets():
    sp500 = pandas_datareader.get_data_yahoo('%5EGSPC',
                                        start=datetime.datetime(2019, 10, 15),
                                        end=datetime.datetime(2019, 10, 19))   # api to get sp500 chart prices
    print(sp500.head())
    df = pd.DataFrame(sp500)

    df.to_csv('sp500-historical.csv')


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
##    getPriceData('sp500.csv')
##    printDF("sp500.csv")

    #make_dataframe_from_stock_tweets()
    #getPriceData('sp500-historical.csv')
    #printDF("sp500-historical.csv")


    ###############################

    polarityList = {}
##    dateRange = [date.strftime('%Y-%m-%d') for date in pd.date_range(start="2019-10-15", end="2019-11-15")] ## upperbound not included
##
##    for date in dateRange:
##        print(date)
##        
##        db = client['spx']
##        mydb = db[date]
##        polarities = []
##
##        for doc in mydb.find():
##            if not doc["historical"]:
##                ts = time.strftime('%Y-%m-%d %H:%M:%S', time.strptime(doc['created_at'],
##                                                                  '%a %b %d %H:%M:%S +0000 %Y'))  # get timestamp from database
##            else: ts = doc["date"]
##            
##            if ts.__contains__(date):
##                tweet = doc['text']
##                filtered = filterTweet(tweet)
##                ss = getPolarity(filtered)
##                # print(filtered, ss)
##                if ss['compound'] != 0.0:
##                    polarities.append(ss['compound'])
##        ##            total_polarity += ss['compound']
##        ##            neg_polarity += ss['neg']
##        ##            pos_polarity += ss['pos']
##        ##            textBlob_pol.append(ss['textblob_polarity'])
##                    #count += 1
##
##
##            #mydb.update_one({"id": doc["id"]}, { '$push' : {'polarity_list' : ss['compound']}}, upsert=True)
##            #mydb.update(document=doc, { '$push' : {'polarity_list' : ss['compound']}}, upsert=True)
##
##        polarityList[date] = sum(polarities)/len(polarities)

    f = 'sp500-historical.csv'
    df = pd.read_csv(f, index_col='Date', parse_dates=True)
    
    dates = df.index.array
    polarityList = df['compound_polarity'].array

    print(dates)
    print(polarityList)

    axes = plt.gca()
    axes.set_ylim([-0.4,0.4])

    #polarityList = {'2020-02-20': 0.0026245117187500247, '2020-02-21': 0.022203057199211045, '2020-02-22': 0.11746189956331884, '2020-02-23': 0.0675799597180262, '2020-02-24': -0.0185952544031311, '2020-02-25': -0.08119766401590471, '2020-02-26': -0.017515340086830686, '2020-02-27': -0.14538179969496676, '2020-02-28': 0.006154630083292493, '2020-02-29': -0.013088311688311692, '2020-03-01': -0.05857078916372196, '2020-03-02': 0.0839724289911853, '2020-03-03': -0.07073632887189299, '2020-03-04': 0.05267408337518842, '2020-03-05': -0.06729975739932059, '2020-03-06': -0.000956620125180545, '2020-03-07': -0.00306628383921247, '2020-03-08': -0.04207881355932195, '2020-03-09': -0.0927196491228072, '2020-03-10': 0.027609550000000045, '2020-03-11': -0.11583041975308613, '2020-03-12': -0.15199861317483893, '2020-03-13': 0.04633336698637039, '2020-03-14': -0.048644279786603364, '2020-03-15': -0.09751837811900156, '2020-03-16': -0.12113731268731252, '2020-03-17': 0.004754031117397438, '2020-03-18': -0.04121448780487802, '2020-03-19': -0.00980536141694593, '2020-03-20': -0.0987696869070208, '2020-03-21': -0.05405679257786614, '2020-03-22': -0.11718664987405489, '2020-03-23': -0.04409216362740274}
    polarityX = list(dates)
    polarityX_np = np.array(polarityX)
    
    polarityY = polarityList
    polarityY_np = np.array(polarityY)

    plt.plot(polarityX, polarityY)

    m, b = np.polyfit(np.array([x for x in range(0,len(polarityX))]), polarityY_np, 1)
    print(m, b)
    
    plt.plot(np.array([x for x in range(0,len(polarityX))]), m*np.array([x for x in range(0,len(polarityX))]) + b)
    
    plt.xlabel('date')
    plt.ylabel('tweet sentiments')
    plt.title('cool graph')
    plt.show()
    
