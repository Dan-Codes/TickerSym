from pymongo import MongoClient
import twitter
import json
import os
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
import nltk

mongoDB_key = os.environ.get('mongodb')  # replace this with the mongoDB API KEY
client = MongoClient("mongodb+srv://dbSPX:"+mongoDB_key+"@cluster0-p4uhp.mongodb.net/test?retryWrites=true&w=majority")


consumer_key = "0wKlzBq85PVd7vycTJoFJhHsM" # os.environ.get('CONSUMER_KEY')
consumer_secret = "LjvPDBRU4uZK9fUk62V1w5MXBNJBwMbDLoK6Rp0B4GUOewGfSz" # os.environ.get('CONSUMER_SECRET')
access_token = "4543045944-fgbvrOgDksuD2M7zjetCspzczn8jZFrDqquV8uO" # os.environ.get('ACCESS_TOKEN')
access_token_secret = "lWT1WosHWQaIKy4BjeIuWfuArFPrChZ1b9kFjhoUDprcp" # os.environ.get('ACCESS_SECRET_TOKEN')

api = twitter.Api(consumer_key=consumer_key,
                  consumer_secret=consumer_secret,
                  access_token_key=access_token,
                  access_token_secret=access_token_secret)
print(api.VerifyCredentials())


class TwitterStreamer():
    """
    Class for streaming and processing live tweets.
    """

    def __init__(self):
        pass

    def stream_tweets(self, fetched_tweets_filename, hash_tag_list):
        # This handles Twitter authentication and the connection to Twitter Streaming API
        listener = StdOutListener(fetched_tweets_filename)
        auth = OAuthHandler(consumer_key, consumer_secret)
        auth.set_access_token(access_token, access_token_secret)
        stream = Stream(auth, listener)

        # This line filter Twitter Streams to capture data by the keywords:
        stream.filter(track=hash_tag_list)


# # # # TWITTER STREAM LISTENER # # # #
class StdOutListener(StreamListener):

    def __init__(self, fetched_tweets_filename):
        self.fetched_tweets_filename = fetched_tweets_filename

    def on_data(self, data):
        try:
            print(data)
            mydb = client["spx"]
            mycol = mydb["tweets"]
            datajson = json.loads(data)
            x = mycol.insert_one(datajson)
            return True
        except BaseException as e:
            print("Error on_data %s" % str(e))
        return True

    def on_error(self, status):
        print(status)


if __name__ == '__main__':
    # Authenticate using config.py and connect to Twitter Streaming API.
    hash_tag_list = ["S&P500", "SP500", "$SPX", "$SPY"]
    fetched_tweets_filename = "tweets.txt"
    db = client.test
    print(db)
    twitter_streamer = TwitterStreamer()
    twitter_streamer.stream_tweets(fetched_tweets_filename, hash_tag_list)

