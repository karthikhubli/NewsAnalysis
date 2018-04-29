
# coding: utf-8

# In[244]:


from newsapi import NewsApiClient
import bs4 as bs
import urllib as urllib
import urllib.request as ur
import pandas as pd
import tldextract
import numpy as np
from textblob import TextBlob
import six
from sklearn.externals import joblib
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB


# In[245]:


#Categories
#technology | buisness | sports
#business| entertainment| general| health |science |sports |technology
TRAINED_MODEL='trainedModel.pkl'
TFIDF_VECT='wordVect.pkl'
nbModel = None
vect= None
fakeDomains=[]
def init():
    global fakeDomains
    if(len(fakeDomains) == 0):
        print('loading')
        data=pd.read_csv('fakeDomain.csv', sep=',',header=0)
        for row_index,row in data.iterrows():
            fakeDomains.append(str(row['Domain']))
    else:
        print(len(fakeDomains))


# In[246]:


def getDomain(url):
    list = tldextract.extract(url)
    domain_name = list.domain + '.' + list.suffix
    return (domain_name)

def getTopHeadLinesByCateg(apiKey,topic,categ):
    newsapi = NewsApiClient(api_key=apiKey)
    if(categ == "" or categ is None):
        categ='general'
    top_headlines = newsapi.get_top_headlines(q=topic,
                                          category=categ,
                                          language='en',
                                          country='us')
    return top_headlines['articles']

def getAll(apiKey,topic,dtFrom,dtTo,pg):
    newsapi = NewsApiClient(api_key=apiKey)
#     print(dtFrom)
#     print(topic)
#     print(dtTo)
    all_articles = newsapi.get_everything(q=topic,
                                      sources='bbc-news,the-verge,bloomberg',
                                      domains='bbc.co.uk,techcrunch.com,cnn.com',
                                      from_param=dtFrom,
                                      to=dtTo,
                                      page_size=40,
                                      language='en',
                                      sort_by='publishedAt',
                                      page=pg)
    return all_articles['articles'],all_articles['totalResults']

def getAllReferencesOnPage(url):
    
    try:
        sauce = ur.urlopen(url).read()
        soup=bs.BeautifulSoup(sauce,'lxml')
        domArr=[]
        for link in soup.find_all('a'):
            if link.has_attr('href'):
                domArr.append(getDomain(link['href']))
        domains=np.array(domArr)
        domains=np.unique(domains)
        return getFakeDomainCounts(domains)
    except urllib.error.URLError as e:
        print(e.reason)
        print(url)
        return 0

def getFakeDomainCounts(domains):
    global fakeDomains
    fakeDomainCount=0
    domainList=domains.tolist()
    for domain in domainList:
        if(domain in fakeDomains):
            fakeDomainCount=fakeDomainCount+1
    return fakeDomainCount
    


# In[247]:


def authenticateText(text):
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


# In[248]:


def prepareHeadline(apiKey):
    headlines=[]
    rowCount=0
    articles=getTopHeadLinesByCateg(apiKey,"","")
    for article in articles:
         articleBlock=[]
         artStr=str(article['description'])
         if(article['description'] is None):
                artStr=article['title']
         descSent=sentAnalysis(article['description'])
         titleSent=sentAnalysis(article['title'])
         fishyDomains=0
         if(article['url'] is not None):
             fishyDomains=getAllReferencesOnPage(str(article['url']))
         articleBlock.append( {'title':article['title']})
         articleBlock.append( {'desc':article['description']})
         articleBlock.append( {'url':article['url']})
         articleBlock.append( {'image':article['urlToImage']})
         articleBlock.append( {'authenticity':str(authenticateText(artStr))})
         articleBlock.append( {'descPol':str(descSent[0])})
         articleBlock.append( {'descSub':str(descSent[1])})
         articleBlock.append( {'titlePol':str(titleSent[0])})
         articleBlock.append( {'titleSubj':str(titleSent[1])})
         articleBlock.append( {'fishyDomain':str(fishyDomains)})
         headlines.append(articleBlock)
         rowCount=rowCount+1
    return headlines,rowCount

def prepareHeadlineByTopic(apiKey,keyword,cat):
    headlines=[]
    rowCount=0
    articles=getTopHeadLinesByCateg(apiKey,keyword,cat)
    for article in articles:
         articleBlock=[]
         artStr=str(article['description'])
         if(article['description'] is None):
                artStr=article['title']
         descSent=sentAnalysis(article['description'])
         titleSent=sentAnalysis(article['title'])
         fishyDomains=0
         if(article['url'] is not None):
             fishyDomains=getAllReferencesOnPage(str(article['url']))
         articleBlock.append( {'title':article['title']})
         articleBlock.append( {'desc':article['description']})
         articleBlock.append( {'url':article['url']})
         articleBlock.append( {'image':article['urlToImage']})
         articleBlock.append( {'authenticity':str(authenticateText(artStr))})
         articleBlock.append( {'descPol':str(descSent[0])})
         articleBlock.append( {'descSub':str(descSent[1])})
         articleBlock.append( {'titlePol':str(titleSent[0])})
         articleBlock.append( {'titleSubj':str(titleSent[1])})
         articleBlock.append( {'fishyDomain':str(fishyDomains)})
         headlines.append(articleBlock)
         rowCount=rowCount+1
    return headlines,rowCount


# In[249]:


init()

