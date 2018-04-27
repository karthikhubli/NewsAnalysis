
# coding: utf-8

# In[48]:


import tldextract
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib
from flask import Flask
from flask_restplus import Api,Resource,fields,reqparse
from textblob import TextBlob
import six
import NewsAPIParser as nparse
import datetime


# In[49]:


TRAINED_MODEL='trainedModel.pkl'
TFIDF_VECT='wordVect.pkl'
nbModel = None
vect= None
app = Flask(__name__)
api=Api(app)


news_parser = reqparse.RequestParser()
news_parser.add_argument('news', required=True, help="Enter the news Article")
news_parser.add_argument('title', required=True, help="Enter the news Title")

apiKey_parser = reqparse.RequestParser()
apiKey_parser.add_argument('apiKey', required=True, help="API key from NewsAPI.org")

topicParam_parser = reqparse.RequestParser()
topicParam_parser.add_argument('apiKey', required=True, help="API key from NewsAPI.org")
topicParam_parser.add_argument('topic', required=True, help="Topic to search")
topicParam_parser.add_argument('category', required=True, help="business| entertainment| general| health |science |sports |technology")

update_parser=reqparse.RequestParser()
update_parser.add_argument('feedback', required=True, help="Enter the news Article")


# In[51]:





# In[52]:


#REST API
@api.route('/credibility')
class Credibility(Resource):
    @api.expect(news_parser)
    def get(self):
        credibility=[]
        args = news_parser.parse_args()
        newsText=args['news']
        newsTitle=args['title']
        score=nparse.authenticateText(newsText)
        sent=nparse.sentAnalysis(newsText)
        titleScore=nparse.authenticateText(newsTitle)
        titleSent=nparse.sentAnalysis(newsTitle)
        credibility.append( {'news_authenticity':str(score)})
        credibility.append( {'news_polarity':sent[0]})
        credibility.append( {'news_subjectivity':sent[1]})
        credibility.append( {'title_authenticity':str(titleScore)})
        credibility.append( {'title_polarity':titleSent[0]})
        credibility.append( {'title_subjectivity':titleSent[1]})
        return credibility
      
    @api.expect(update_parser)
    def post(self):
        args = update_parser.parse_args()
        fb=args['feedback']
        print('--------------------feedback=',fb)
        return {'result' : 'Feedback received'}


# In[53]:


@api.route('/headlines')
class NewsHeadlines(Resource):
    @api.expect(apiKey_parser)
    def get(self):
        args = apiKey_parser.parse_args()
        apiKey=args['apiKey']
        articles=nparse.prepareHeadline(apiKey)
        topHeadline=[]
        topHeadline.append({'type':'verity-story top headlines'})
        topHeadline.append({'updated':str(datetime.datetime.now().isoformat())})
        topHeadline.append({'newscount':str(articles[1])})
        topHeadline.append({'articleBlock':articles[0]})
        return topHeadline
 


# In[ ]:


@api.route('/newsbytopic')
class NewsByTopic(Resource):
    @api.expect(topicParam_parser)
    def get(self):
        args = topicParam_parser.parse_args()
        apiKey=args['apiKey']
        topic=args['topic']
        category=args['category']
        articles=nparse.prepareHeadlineByTopic(apiKey,topic,category)
        topHeadline=[]
        topHeadline.append({'type':'verity-story top headlines'})
        topHeadline.append({'updated':str(datetime.datetime.now().isoformat())})
        topHeadline.append({'newscount':str(articles[1])})
        topHeadline.append({'articleBlock':articles[0]})
        return topHeadline


# In[54]:


if(__name__=='__main__'):
    app.run(debug=True)

