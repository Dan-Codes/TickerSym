import json
from datetime import datetime, timedelta
from statistics import mean

import pandas as pd
from newsapi import NewsApiClient
from nltk.sentiment import SentimentIntensityAnalyzer

import nltkAnalysis

import stock_market_lexicon


def formatTime(now):
    year = '{:02d}'.format(now.year)
    month = '{:02d}'.format(now.month)
    day = '{:02d}'.format(now.day)
    hour = '{:02d}'.format(now.hour)
    minute = '{:02d}'.format(now.minute)
    day_month_year = '{}-{}-{}'.format(year, month, day)
    return day_month_year


class newsAPI:

    def getNews(self):
        newsapi = NewsApiClient(api_key='6cd77e4a618d4a75907ba6714b850a23')

        # /v2/top-headlines
        # top_headlines = newsapi.get_top_headlines(qintitle='Stock',
        #                                           category='business',
        #                                           language='en',
        #                                           country='us')

        # /v2/everything
        date_14_days_ago = formatTime(datetime.now() - timedelta(days=14))
        today = formatTime(datetime.now())
        all_articles = newsapi.get_everything(q='s&p 500',
                                              from_param=date_14_days_ago,
                                              to=today,
                                              language='en',
                                              sort_by='relevancy'
                                              )

        # /v2/sources
        sources = newsapi.get_sources()

        print(json.dumps(all_articles, indent=1))
        newsAPI().saveCSV(all_articles)
        return all_articles

    def processHeadlines(self):
        df = pd.read_csv('newsHeadlines.csv', index_col='publishedAt', parse_dates=True)
        compound = []
        for title in df.index.array:
            headline = df.at[title, 'title']
            headline = nltkAnalysis.filterTweet(headline)
            pol = nltkAnalysis.getPolarity(headline)
            print(headline, pol)
            compound.append(pol['compound'])
            df.at[title, 'polarity'] = pol['compound']
        print("average polarity=", mean(compound))
        df.to_csv('newsHeadlines.csv')

    @staticmethod
    def saveCSV(articles):
        articles = articles['articles']
        df = pd.DataFrame(articles)
        df.to_csv('newsHeadlines.csv')
        df = pd.read_csv('newsHeadlines.csv', index_col='publishedAt', parse_dates=True)
        df.to_csv('newsHeadlines.csv')


if __name__ == '__main__':
    print(json.dumps(newsAPI().getNews(), indent=1))
    newsAPI().processHeadlines()
