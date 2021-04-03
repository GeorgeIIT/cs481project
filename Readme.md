#### Tweets From U.S. Congress Members 

The file 'party_tweets.csv' contains tweets and retweets from members of the US Congress. Each tweet is labeled with the congress member's party affiliation. The partys are Democrats and Republicans, Members of a third party were not included. Tweets were collected by pulling the last 40 tweets (retweets included) from each congress member's user timeline on 4/1/21. The goal for this repository is to predict the political party of a legislator based on the content they share online. It should be noted that these are official twitter accounts so the vast majority of the content posted is relevant to the user's political career. Models trained with this dataset may not be effective for predicting the party affiliation of non-political content, or political content from perspectives outside of the United States. 
* Total tweets: 21181
* Democrat Tweets (D): 10965
* Republican Tweets (R): 10225
* Ratio: 52:48 D:R 

The code used for collecting the twitter data is in 'Get_congress_tw_data.py'. If you wish to run it, you will need to provide your own twitter authentication tokens. Also be warned that it takes several hours to run because of the rate limit for twitter API calls.

Congress member twitter handles were found on [Triage Cancer's](https://triagecancer.org/congressional-social-media) website.