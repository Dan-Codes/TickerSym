# TickerSym
Sentiment Analysis of S&P 500 (SPX) in relation to Trading Price


# Installation and Run
    pip install -r requirements.txt
    python nltkAnalysis.py

#miningTweets.py
Uses the streamingAPI to place tweets about the S&P500 into the MongoDB

#nltkAnalysis.py
Takes the data from the mongoDB and gets the polarity of each tweet and places the information on a dataframe

#stock_market_lexicon.py
improved lexicon for Vader NLTK used for analyzing investing tweets

#oldTweets.py
mines old tweets 


