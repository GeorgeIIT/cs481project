# -*- coding: utf-8 -*-
"""
Created on Sun Feb 21 20:28:18 2021

@author: George Schaefer

test dataset: members of congress and their respective parties 
Congress social media handles: https://triagecancer.org/congressional-social-media
info from this website is stored in congress_sm_handles.txt
    
"""
import config #this is where I keep my twitter dev keys
import tweepy
import time
import pandas as pd
import re #regular expression library


def handle_handler():
    """
    this code is meant to format the information from the text document into a .csv file
    and return the name of that csv file.
    """
    
    #read each line from the website
    with open("congress_sm_handles.txt","r",encoding='utf-8') as read_handles:
        raw_lines = read_handles.readlines()
    
    #csv file to write the formatted lines into
    output_filename = "congress_sm_handles.csv"
    write_handles = open(output_filename,"w")
    
    #remove commas on names and replace tabs with commas to delimit info
    for line in raw_lines:
        line = line.replace(',','')
        line = line.replace('\t',',')
        write_handles.write(line)
    write_handles.close()
    
    print("formatted twitter handles and placed in ", output_filename)
    
    return output_filename
    

"""
scrape_tweets collects the tweets from each congress member and places the min a single .csv file with
three rows: idx, tweet text, party
"""
def scrape_tweets(handles,CK,CS,BT,AT,AS):
    #access twitter api
    auth = tweepy.OAuthHandler(CK,CS)
    auth.set_access_token(AT,AS)
    api = tweepy.API(auth)
    
    #read in all of the congress twitter handles 
    congress_handles = pd.read_csv(handles,encoding='latin-1')
    congress_handles = congress_handles[['Name','Party','Twitter']]
    
    output_filename = "party_tweets.csv"
    output_file = open(output_filename,"w",encoding='utf-8')
    
    tweet_counter = 0
    #iterate over all twitter handles and scrape tweets
    for index, row in congress_handles.iterrows():
        valid_member = is_valid_member(row['Twitter'], row['Party'])
        if valid_member:
            member_handle = row['Twitter'].replace('@','')
            member_party = row['Party']
            #scrape the tweets from congress member's handle, store in .csv file
            tweet_count = 40
            try:
                tweets = tweepy.Cursor(api.user_timeline,id=member_handle,tweet_mode='extended').items(tweet_count)
                tweet_list = [[tweet.retweeted_status.full_text if tweet.full_text.startswith("RT @") else tweet.full_text] for tweet in tweets]
                for item in tweet_list:
                    raw_text = str(item[0])
                    tweet_text = format_tweet(raw_text)
                    line_entry = str(tweet_counter) + "|" + tweet_text + "|" + member_party + "\n"
                    output_file.write(line_entry)
                    tweet_counter += 1
                    print(line_entry)
                #sleep for 13 seconds to avoid the twitter 429 error
                time.sleep(52)
            except BaseException as e:
                print('failed on_status,',str(e))
                time.sleep(3)
                
    output_file.close()
    print("scraped tweets from all congress members")
    return output_filename
    
#exclude members of congress that don't have twitter or are members of a third party
def is_valid_member(twitter_handle, party):
    valid = type(twitter_handle) == type("string") #handle exists
    valid = valid and type(party) == type("string") #party exists
    valid = valid and ((party == 'R') or (party == 'D')) #not a third party member
    return valid

"""
format_tweet() takes the raw text of the tweet and formats as follows:
    -remove punctuation and special chars
    -convert chars to lower case
    -remove web links
"""
def format_tweet(raw_text):
    tweet_text = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '<webLink>', raw_text) #replace links with 'webLink'
    #tweet_text = re.sub(r'[^a-zA-Z0-9_\s]+','',tweet_text) #remove special chars
    tweet_text = tweet_text.replace('\n',' ')
    tweet_text = tweet_text.replace('\t',' ')
    tweet_text = tweet_text.replace('\r',' ')
    tweet_text = tweet_text.replace('&amp;', '&')
    tweet_text = tweet_text.replace('|', ' ') #get rid of any pipes to avoid issues with delimiter
    #tweet_text = tweet_text.lower() #convert all chars to lowercase
    return tweet_text

#get all of the twitter handles and party affiliations into a .csv file
handles_file = handle_handler()

tweets_file = scrape_tweets(handles_file,config.CONSUMER_KEY,config.CONSUMER_SECRET,config.BEARER_TOKEN,config.ACCESS_TOKEN,config.ACCESS_SECRET)