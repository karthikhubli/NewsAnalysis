
# coding: utf-8

# In[120]:


import tldextract
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib
newsText='Maria Sharapova overpowered Polandâ€™s Magda Linette to enter the quarter-finals of the Tianjin Open on Thursday and is targeting a first title since returning from a 15-month doping ban.'


# In[121]:


def loadData(filePath):
    df=pd.read_csv(filePath, sep=',',header=0)
    data=clubAdditionalData(df)
    #data.head(15)
    return df

def clubAdditionalData(df):
    data=pd.read_csv('C:/Users/sneha/Desktop/FakingNews/Dataset/fake.csv', sep=',',header=0)
    newData =data[['text','spam_score']].copy()
    newData['Label']=1
    newData['Label']=newData['spam_score'].apply(normalizeSpamScore)
    newData.drop(['spam_score'],axis=1)
    print('?')
    newData.head(5)
    #newData.columns = ['Body', 'Label']
    #appended=df.appent(newData)
    return 0

def normalizeSpamScore(score):
    if(score < 0.75):
        return 0
    else:
        return 1


# In[122]:


#NLP with MN_NaievBais
def trainModel(df,x_train,x_test,y_train,y_test):
    cv=TfidfVectorizer(min_df=1,stop_words='english')
    x_trainCV=cv.fit_transform(x_train)
    x_testCV=cv.transform(x_test)
    mnb_model=MultinomialNB()
    mnb_model.fit(x_trainCV,y_train)
    print('Done Training')
    return mnb_model,x_testCV,cv


# In[123]:


def testModel(model,x_test,y_test):
    result=model.predict(x_test)
    rms=np.mean((result-y_test)**2)
    accuracy=model.score(x_test,y_test)
    print('RMS Error Value-',rms)
    print('Accuracy-',accuracy)


# In[124]:


def serializeModel(trModel):
    joblib.dump(trModel, 'trainedModel.pkl') 
    return 'serialized trained mODEL'

def serializeWordVect(wordVec):
    joblib.dump(wordVec, 'wordVect.pkl') 
    return 'serialized word vector'


# In[125]:


def authenticateNews(text,model,vect):
    text_arr=[text]
    tfidfVect=vect.transform(text_arr)
    print('Prediction')
    result=model.predict(tfidfVect)[0]
    if(result ==0):
        print('Alert:The article is not authentic')
    else:
        print('The article is authentic!!!!!!')
    return result


# In[126]:


def mainProg():
    dataF=loadData('data.csv')
    df_x=dataF['Body'].values.astype('U')
    df_y=dataF['Label'].values.astype('int')
    x_train,x_test,y_train,y_test= train_test_split(df_x,df_y,test_size=0.2,random_state=4)
    trModel,x_testCV,countV=trainModel(dataF,x_train,x_test,y_train,y_test)
    testModel(trModel,x_testCV,y_test)
    authenticateNews(newsText,trModel,countV);
    


# In[127]:


mainProg();

