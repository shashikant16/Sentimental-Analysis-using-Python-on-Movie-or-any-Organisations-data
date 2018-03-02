#import regex
import re

#start process_tweet
def processTweet(tweet):
    # process the tweets

    #Convert to lower case
    tweet = tweet.lower()
    #Convert www.* or https?://* to URL
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL',tweet)
    #Convert @username to AT_USER
    tweet = re.sub('@[^\s]+','AT_USER',tweet)
    #Remove additional white spaces
    tweet = re.sub('[\s]+', ' ', tweet)
    #Replace #word with word
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
    #trim
    tweet = tweet.strip('\'"')
    return tweet

#Read the tweets one by one and process it
fp = open('data/sampleTweets.txt', 'r')
line = fp.readline()

while line:
    processedTweet = processTweet(line)
    print processedTweet
    line = fp.readline()
fp.close()

#Filtering tweet words (for feature vector)

#initialize stopWords
stopWords = []

#start replaceTwoOrMore
def replaceTwoOrMore(s):
    #look for 2 or more repetitions of character and replace with the character itself
    pattern = re.compile(r"(.)\1{1,}", re.DOTALL)
    return pattern.sub(r"\1\1", s)

#start getStopWordList
def getStopWordList(stopWordListFileName):
    #read the stopwords file and build a list
    stopWords = []
    stopWords.append('AT_USER')
    stopWords.append('URL')

    fp = open(stopWordListFileName, 'r')
    line = fp.readline()
    while line:
        word = line.strip()
        stopWords.append(word)
        line = fp.readline()
    fp.close()
    return stopWords

#start getfeatureVector
def getFeatureVector(tweet):
    featureVector = []
    #split tweet into words
    words = tweet.split()
    for w in words:
        #replace two or more with two occurrences
        w = replaceTwoOrMore(w)
        #strip punctuation
        w = w.strip('\'"?,.')
        #check if the word stats with an alphabet
        val = re.search(r"^[a-zA-Z][a-zA-Z0-9]*$", w)
        #ignore if it is a stop word
        if(w in stopWords or val is None):
            continue
        else:
            featureVector.append(w.lower())
    return featureVector

#Read the tweets one by one and process it
fp = open('data/sampleTweets.txt', 'r')
line = fp.readline()

st = open('data/feature_list/stopwords.txt', 'r')
stopWords = getStopWordList('data/feature_list/stopwords.txt')

while line:
    processedTweet = processTweet(line)
    featureVector = getFeatureVector(processedTweet)
    print featureVector
    line = fp.readline()
fp.close()

#Feature Extraction

#Read the tweets one by one and process it
inpTweets = csv.reader(open('data/sampleTweets.csv', 'rb'), delimiter=',', quotechar='|')
tweets = []
for row in inpTweets:
    sentiment = row[0]
    tweet = row[1]
    processedTweet = processTweet(tweet)
    featureVector = getFeatureVector(processedTweet, stopWords)
    tweets.append((featureVector, sentiment));
    
#Tweets Variable
tweets = [(['hey', 'cici', 'luv', 'mixtape', 'drop', 'soon', 'fantasy', 'ride'], 'positive'),
           (['heard', 'congrats'], 'positive'),
           (['ncaa', 'franklin', 'wild'], 'positive'),
           (['share', 'jokes', 'quotes', 'music', 'photos', 'news', 'articles', 'facebook', 'twitter'], 'neutral'),
           (['night', 'twitter', 'thelegionofthefallen', 'cimes', 'awfully'], 'neutral'),
           (['finished', 'mi', 'run', 'pace', 'gps', 'nikeplus', 'makeitcount'], 'neutral'),
           (['disappointing', 'day', 'attended', 'car', 'boot', 'sale', 'raise', 'funds', 'sanctuary',
             'total', 'entry', 'fee', 'sigh'], 'negative'),
           (['taking', 'irish', 'car', 'bombs', 'strange', 'australian', 'women', 'drink', 'head',
             'hurts'], 'negative'),
           (['bloodwork', 'arm', 'hurts'], 'negative')]

#Feature List

featureList = ['hey', 'cici', 'luv', 'mixtape', 'drop', 'soon', 'fantasy', 'ride', 'heard',
'congrats', 'ncaa', 'franklin', 'wild', 'share', 'jokes', 'quotes', 'music', 'photos', 'news',
'articles', 'facebook', 'twitter', 'night', 'twitter', 'thelegionofthefallen', 'cimes', 'awfully',
'finished', 'mi', 'run', 'pace', 'gps', 'nikeplus', 'makeitcount', 'disappointing', 'day', 'attended',
'car', 'boot', 'sale', 'raise', 'funds', 'sanctuary', 'total', 'entry', 'fee', 'sigh', 'taking',
'irish', 'car', 'bombs', 'strange', 'australian', 'women', 'drink', 'head', 'hurts', 'bloodwork',
'arm', 'hurts']

#Extract Features Method

#start extract_features
def extract_features(tweet):
    tweet_words = set(tweet)
    features = {}
    for word in featureList:
        features['contains(%s)' % word] = (word in tweet_words)
    return features

#Output of Extract Features

{
    'contains(arm)': True,             #notice this
    'contains(articles)': False,
    'contains(attended)': False,
    'contains(australian)': False,
    'contains(awfully)': False,
    'contains(bloodwork)': True,       #notice this
    'contains(bombs)': False,
    'contains(cici)': False,
    .....
    'contains(head)': False,
    'contains(heard)': False,
    'contains(hey)': False,
    'contains(hurts)': True,           #notice this
    .....
    'contains(irish)': False,
    'contains(jokes)': False,
    .....
    'contains(women)': False
}

# Bulk Extraction of Features

#Read the tweets one by one and process it
inpTweets = csv.reader(open('data/sampleTweets.csv', 'rb'), delimiter=',', quotechar='|')
stopWords = getStopWordList('data/feature_list/stopwords.txt')
featureList = []

# Get tweet words
tweets = []
for row in inpTweets:
    sentiment = row[0]
    tweet = row[1]
    processedTweet = processTweet(tweet)
    featureVector = getFeatureVector(processedTweet, stopWords)
    featureList.extend(featureVector)
    tweets.append((featureVector, sentiment));

# Remove featureList duplicates
featureList = list(set(featureList))

# Extract feature vector for all tweets in one shote
training_set = nltk.classify.util.apply_features(extract_features, tweets)

#Naive Bayes Classifier

# Train the classifier
NBClassifier = nltk.NaiveBayesClassifier.train(training_set)

# Test the classifier
testTweet = 'Congrats @ravikiranj, i heard you wrote a new tech post on sentiment analysis'
processedTestTweet = processTweet(testTweet)
print NBClassifier.classify(extract_features(getFeatureVector(processedTestTweet)))

#Informative Features
# print informative features about the classifier
print NBClassifier.show_most_informative_features(10)

testTweet = 'I am so badly hurt'
processedTestTweet = processTweet(testTweet)
print NBClassifier.classify(extract_features(getFeatureVector(processedTestTweet)))

#Maximum Entropy Classifier
#Max Entropy Classifier
MaxEntClassifier = nltk.classify.maxent.MaxentClassifier.train(training_set, 'GIS', trace=3, \
                    encoding=None, labels=None, sparse=True, gaussian_prior_sigma=0, max_iter = 10)
testTweet = 'Congrats @ravikiranj, i heard you wrote a new tech post on sentiment analysis'
processedTestTweet = processTweet(testTweet)
print MaxEntClassifier.classify(extract_features(getFeatureVector(processedTestTweet)))

#print informative features
print MaxEntClassifier.show_most_informative_features(10)

# Support Vector Machines

import svm
from svmutil import *

#training data
labels = [0, 1, 1, 2]
samples = [[0, 1, 0], [1, 1, 1], [1, 1, 0], [0, 0, 0]]

#SVM params
param = svm_parameter()
param.C = 10
param.kernel_type = LINEAR
#instantiate the problem
problem = svm_problem(labels, samples)
#train the model
model = svm_train(problem, param)
# saved model can be loaded as below
#model = svm_load_model('model_file')

#save the model
svm_save_model('model_file', model)

#test data
test_data = [[0, 1, 1], [1, 0, 1]]
#predict the labels
p_labels, p_accs, p_vals = svm_predict([0]*len(test_data), test_data, model)
print p_labels

def getSVMFeatureVectorAndLabels(tweets, featureList):
    sortedFeatures = sorted(featureList)
    map = {}
    feature_vector = []
    labels = []
    for t in tweets:
        label = 0
        map = {}
        #Initialize empty map
        for w in sortedFeatures:
            map[w] = 0

        tweet_words = t[0]
        tweet_opinion = t[1]
        #Fill the map
        for word in tweet_words:
            #process the word (remove repetitions and punctuations)
            word = replaceTwoOrMore(word)
            word = word.strip('\'"?,.')
            #set map[word] to 1 if word exists
            if word in map:
                map[word] = 1
        #end for loop
        values = map.values()
        feature_vector.append(values)
        if(tweet_opinion == 'positive'):
            label = 0
        elif(tweet_opinion == 'negative'):
            label = 1
        elif(tweet_opinion == 'neutral'):
            label = 2
        labels.append(label)
    #return the list of feature_vector and labels
    return {'feature_vector' : feature_vector, 'labels': labels}
#end

#Train the classifier
result = getSVMFeatureVectorandLabels(tweets, featureList)
problem = svm_problem(result['labels'], result['feature_vector'])
#'-q' option suppress console output
param = svm_parameter('-q')
param.kernel_type = LINEAR
classifier = svm_train(problem, param)
svm_save_model(classifierDumpFile, classifier)

#Test the classifier
test_feature_vector = getSVMFeatureVector(test_tweets, featureList)
#p_labels contains the final labeling result
p_labels, p_accs, p_vals = svm_predict([0] * len(test_feature_vector),test_feature_vector, classifier)

#get_twitter_data.py

import argparse
import urllib
import json
import os
import oauth2

class TwitterData:
    def parse_config(self):
        config = {}
        # from file args
        if os.path.exists('config.json'):
            with open('config.json') as f:
                config.update(json.load(f))
        else:
            # may be from command line
            parser = argparse.ArgumentParser()

            parser.add_argument('-ck', '--consumer_key', default=None, help='Your developper `Consumer Key`')
            parser.add_argument('-cs', '--consumer_secret', default=None, help='Your developper `Consumer Secret`')
            parser.add_argument('-at', '--access_token', default=None, help='A client `Access Token`')
            parser.add_argument('-ats', '--access_token_secret', default=None, help='A client `Access Token Secret`')

            args_ = parser.parse_args()
            def val(key):
                return config.get(key)\
                    or getattr(args_, key)\
                    or raw_input('Your developper `%s`: ' % key)
            config.update({
                'consumer_key': val('consumer_key'),
                'consumer_secret': val('consumer_secret'),
                'access_token': val('access_token'),
                'access_token_secret': val('access_token_secret'),
            })
        # should have something now
        return config
    #end

    def oauth_req(self, url, http_method="GET", post_body=None,
                  http_headers=None):
        config = self.parse_config()
        consumer = oauth2.Consumer(key=config.get('consumer_key'), secret=config.get('consumer_secret'))
        token = oauth2.Token(key=config.get('access_token'), secret=config.get('access_token_secret'))
        client = oauth2.Client(consumer, token)

        resp, content = client.request(
            url,
            method=http_method,
            body=post_body or '',
            headers=http_headers
        )
        return content
    #end

    #start getTwitterData
    def getData(self, keyword, params = {}):
        maxTweets = 50
        url = 'https://api.twitter.com/1.1/search/tweets.json?'
        data = {'q': keyword, 'lang': 'en', 'result_type': 'recent', 'count': maxTweets, 'include_entities': 0}

        #Add if additional params are passed
        if params:
            for key, value in params.iteritems():
                data[key] = value

        url += urllib.urlencode(data)

        response = self.oauth_req(url)
        jsonData = json.loads(response)
        tweets = []
        if 'errors' in jsonData:
            print "API Error"
            print jsonData['errors']
        else:
            for item in jsonData['statuses']:
                tweets.append(item['text'])
        return tweets
    #end
