
# coding: utf-8

# In[17]:


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
print('Done importing')


# In[18]:


TRAINED_MODEL='trainedModel.pkl'
TFIDF_VECT='wordVect.pkl'
nbModel = None
vect= None
app = Flask(__name__)
api=Api(app)

#a_language = api.model('Language', {'language' : fields.String('The language.')})
news_format=api.model('News', { 'news' : fields.String('Vietnam ')})
#credibility=[]

news_parser = reqparse.RequestParser()
news_parser.add_argument('news', required=True, help="Enter the news Article")
news_parser.add_argument('title')


print('Done with global variables')


# In[21]:


def authenticateText(text):
    print(text)
    global vect
    global nbModel
    if(vect is None or nbModel is None):
        print('loading discrecetly')
        nbModel = joblib.load(TRAINED_MODEL)
        vect= joblib.load(TFIDF_VECT)
    text_arr=[text]
    tfidfVect=vect.transform(text_arr)
    result=nbModel.predict(tfidfVect)[0]
    return result

def sentAnalysis(text):
    if isinstance(text, six.string_types):
        analysis=TextBlob(text)
        return analysis.sentiment[0],analysis.sentiment[1]
    else:
        return -100.0,-100.0


# In[22]:


#REST API
@api.route('/credibility')
#@api.doc(params={'news': 'the news article here'})
@api.expect(news_parser)
class Credibility(Resource):
    def get(self):
        credibility=[]
        args = news_parser.parse_args()
        newsText=args['news']
        newsTitle=args['title']
        print('-----------------------------',newsTitle)
        score=authenticateText(newsText)
        sent=sentAnalysis(newsText)
        titleScore=authenticateText(newsTitle)
        titleSent=sentAnalysis(newsTitle)
        credibility.append( {'news_authenticity':str(score)})
        credibility.append( {'news_polarity':sent[0]})
        credibility.append( {'news_subjectivity':sent[1]})
        credibility.append( {'title_authenticity':str(titleScore)})
        credibility.append( {'title_polarity':titleSent[0]})
        credibility.append( {'title_subjectivity':titleSent[1]})
        
        return credibility


# In[23]:


if(__name__=='__main__'):
    app.run(debug=True)

