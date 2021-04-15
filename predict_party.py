# -*- coding: utf-8 -*-
"""
Created on Sun Feb 21 20:11:01 2021

@author: Joji
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import string

"""
Import the tweets here once they're formatted into a table with these columns:
idx,text,party flag
"""

data = pd.read_csv("party_tweets.csv",sep='|',engine='python',encoding='utf-8',quoting=3)

#check empty cells (should be 0 for this dataset)
print(data.isnull().sum())

#drop duplicate entries, this data set happens to have 10
data = data.drop_duplicates(keep="first")
print(data.shape)

#reset the index numbers since things got deleted
data = data.reset_index(drop=True)

#check the balance of the data
print(data.party.value_counts())

#take a look at the length of the tweets, this also adds a new column to the dataset
data['length'] = data['tweet'].apply(len)
print(data['length'].describe())

#get a visualization of the length distribution by party
data.hist(column='length',by='party',bins=30,figsize=(12,4),rwidth=0.9)
plt.show()


'''
note: a lot of the tweets max out at 280 charechters as we would expect, but a few outliers are longer (max is 735).
These are retweet chains, which is why they are able to go beyond the maximum length of a tweet. Consider dropping them
during preprocessing. 
'''

#some functions for cleaning up the data, these come from lect 13 on https://github.com/iit-cs481/main
from tqdm import tqdm
import re
from bs4 import BeautifulSoup
#from nltk.stem import PorterStemmer

def decontracted(phrase):
    """
    We first define a function to expand the contracted phrase into normal words
    """
    # specific
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase) # prime 
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    
    return phrase

def clean_text(df):
    """
    Clean the review texts
    """
    cleaned_review = []
    
    for tweet_text in tqdm(df['tweet']):
        
        # expand the contracted words
        tweet_text = decontracted(tweet_text)
        
        #remove html tags
        tweet_text = BeautifulSoup(tweet_text, 'lxml').get_text().strip() # re.sub(r'<.*?>', '', text)
        
        #remove non-alphabetic characters
        tweet_text = re.sub("[^a-zA-Z]"," ", tweet_text)
    
        #the urls from tweets were replaced with '<weblink>', get rid of them here
        tweet_text = tweet_text.replace('<weblink>', '')
        
        #removing punctutation, string.punctuation in python consists of !"#$%&\'()*+,-./:;<=>?@[\\]^_{|}~`
        tweet_text = tweet_text.translate(str.maketrans('', '', string.punctuation))
        # ''.join([char for char in movie_text_data if char not in string.punctuation])
        
        #remove emails
        tweet_text = re.sub(r"(^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$)", '', tweet_text)
        
        cleaned_review.append(tweet_text)

    return cleaned_review

def convert_party_to_number(df):
    converted_label = []
    for numeric_party in tqdm(df['party']):
        if(numeric_party == 'D'):
            numeric_party = 0
        else:
            numeric_party = 1
        converted_label.append(numeric_party)
    return converted_label

#use the new functions of the data
print("\ncleaning tweets")
data['cleaned_tweet'] = clean_text(data)

print("\nconverting party 'D' and 'R' to 0 and 1")
data['numeric_party'] = convert_party_to_number(data)

#check out the one of the tweets before and after cleaning
#print(data['tweet'][11],'\n')
#print(data['cleaned_tweet'][11],'\n')

'''
note: look at tweet 11, they're talking about a bill or something and reffering to it as 'S.1'.
would it be better to leave in stuff like that so we can tell when legislators are all talking about the same bill?

pros:
- one party may talk about legislation in their tweets more than the other party
- historically significant legislation or court rulings may be talked about long after they were decided on

cons:
-this could be time sensitive, S.1 may have been relevant when this data was collected, but will it be talked about in the future?
-i suspect that people tend to reffer to significant historical legislation and court rulings by nicknames or shortened 
versions of the full name. e.g. nobody says '410 U.S. 113' or 'Jane Roe, et al. v. Henry Wade' when they want to talk 
about the landmark supreme court ruling, they just call it 'roe v. wade'. shortened nicknames may be far more common in the
case of tweets where users are limited to 280 charecters, and are trying to reach their constituants rather than other 
legislators who know the legal names of legislation and rulings. 
'''

#stop words to ignore in the model
'''
note: play around with the stop words based on the results we get, some stop words might be relevant to
political discussion

also the contractions here are redundant because i already expanded them with decontracted() so I might remove them later
'''
stopwords= set(['br', 'the', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",\
            "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', \
            'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their',\
            'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', \
            'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', \
            'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', \
            'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',\
            'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',\
            'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',\
            'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very', \
            's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', \
            've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn',\
            "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',\
            "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", \
            'won', "won't", 'wouldn', "wouldn't","no","nor","not"])
    
#tfidfVectorizing
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(lowercase=True,stop_words=stopwords,norm='l2',use_idf=True,smooth_idf=True,sublinear_tf=False)
X = vectorizer.fit_transform(data['cleaned_tweet'])
Y = data['numeric_party']
'''
#take a look at the vectors
print(X.shape)
print(Y.shape)
#print(X.toarray())
#print out the words represented by the vector to check them out
#print(vectorizer.get_feature_names())
'''

from sklearn.model_selection import train_test_split

train_idx, test_idx = train_test_split(np.arange(data.shape[0]), test_size=0.2, 
                                       shuffle=True, random_state=42)

#train/test split the data
X_train = X[train_idx]
Y_train = Y[train_idx]
X_test = X[test_idx]
Y_test = Y[test_idx]

print("Training data: X_train : {}, Y_train : {}".format(X_train.shape, Y_train.shape))
print("Testing data: X_test : {}, Y_test : {}".format(X_test.shape, Y_test.shape))

from sklearn.linear_model import LogisticRegression

#fit the logistic regression
lr_clf = LogisticRegression()
lr_clf.fit(X_train, Y_train)
y_pred_test = lr_clf.predict(X_test)
y_predprob_test = lr_clf.predict_proba(X_test)

#check out how well the model performed
from sklearn.metrics import classification_report,confusion_matrix

print(classification_report(Y_test, y_pred_test))

from sklearn.model_selection import cross_val_score

scores = cross_val_score(lr_clf, X, Y, cv=5, scoring='precision')

from sklearn.metrics import roc_curve, auc

fpr, tpr, thresholds = roc_curve(y_true = Y_test, y_score = y_predprob_test[:,1], pos_label=1)
roc_auc = auc(fpr, tpr) # area under ROC curve

plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Republican Rate')
plt.ylabel('True Republican Rate')
plt.title('ROC (Receiver operating characteristic) curve')
plt.legend(loc="lower right")
plt.show()

feature_to_coef = {word: float("%.3f" % coef) for word, coef in zip(vectorizer.get_feature_names(), lr_clf.coef_[0])}

print("\nTop party 1 features:")
for item in (sorted(feature_to_coef.items(), key=lambda x: x[1], reverse=True)[:15]):
    print(item)

# most of the words are reliable evidence of indicating negative sentiments
print("\nTop party 0 features:")
for item in (sorted(feature_to_coef.items(), key=lambda x: x[1], reverse=False)[:15]):
    print(item)
    
    
'''
bel thinks:
    pary 0 is democrats and party 1 is republicans 
    party 1 has to be republican bc of 'border' 'democrats' 'china' 'great' 'bidenbordercrisis' and 'god' 
    
    party 0 'stopasianhate' 'equal' 'transgender' don't seem like things republicans talk about
    party 0 seems to talk about a more diverse range of topics


'''