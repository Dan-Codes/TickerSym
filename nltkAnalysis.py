import nltk
import miningTweets as mt
from pymongo import MongoClient
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
import unicodedata
sid = SentimentIntensityAnalyzer()
de_token = TreebankWordDetokenizer()
import time

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


if __name__ == '__main__':
    # db = client['spx']
    # mydb = db['tweets']
    # for doc in mydb.find():
    #     ts = time.strftime('%Y-%m-%d %H:%M:%S', time.strptime(doc['created_at'], '%a %b %d %H:%M:%S +0000 %Y'))
    #     if ts.__contains__("2020-04-21"):
    #         tweet = doc['text']
    #         word_token = word_tokenize(tweet)
    #         filtered = filterWords(word_token)
    #         print(ss['compound'])
    sentence = "What an absolute tragedy"
    word_token = word_tokenize(sentence)
    filtered = filterWords(word_token)
    sentence = de_token.detokenize(filtered)
    print(sentence)
    ss = sid.polarity_scores(sentence)
    analysis = TextBlob(sentence)
    print(analysis.sentiment.polarity)
