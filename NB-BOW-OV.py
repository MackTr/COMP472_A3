from collections import Counter
import numpy as np

trainingData = "covid_training.tsv"
testData = "covid_test_public.tsv"
lolol = "test.tsv"

tweets = []
vocabulary = {}
features = {'q1': {'yes': {'totalWords': 0, 'probability': 0}, 'no': {'totalWords': 0, 'probability': 0}}}

smoothingValue = 0.01
logBase = 10

def openFileTrainingSet():
    with open(lolol, encoding="utf8") as file:
        next(file)

        for line in file:
            # split line into list of strings and lowercase
            data = line.lower().split()
            # remove first (tweetID) and last 6 indexes
            label = data[-7:-6]
            content = data[1:-7]
            tweets.append({"content": content, "label" : label})

def getVocabulary():

    for tweet in tweets:
        for word in tweet['content']:
            if word not in vocabulary:
                vocabulary[word] = {'frequency_q1_yes': 0, 'frequency_q1_no': 0, 'p_q1_yes': 0, 'p_q1_no': 0}

def getTotalAndFrequencyOfWordsOfQ1():

    for tweet in tweets:
        if tweet['label'] == ['yes']:
            features['q1']['yes']['totalWords'] += len(tweet['content'])
            for word in tweet['content']:
                vocabulary[word]['frequency_q1_yes'] += 1
        else:
            features['q1']['no']['totalWords'] += len(tweet['content'])
            for word in tweet['content']:
                vocabulary[word]['frequency_q1_no'] += 1

def getProbabilityForEachWordQ1():

    smoothingOfDenominator = smoothingValue * len(vocabulary)

    for word in vocabulary:
        vocabulary[word]['p_q1_yes'] = (vocabulary[word]['frequency_q1_yes'] + smoothingValue) / (features['q1']['yes']['totalWords'] + smoothingValue)
        vocabulary[word]['p_q1_no'] = (vocabulary[word]['frequency_q1_no'] + smoothingValue) / (features['q1']['no']['totalWords'] + smoothingValue)

def getProbabiltyOfQ1():
    numberOfYes = 0
    numberOfNo = 0

    for tweet in tweets:
        if tweet['label'] == ['yes']:
            numberOfYes += 1
        else:
            numberOfNo += 1

    features['q1']['yes']['probability'] = float(numberOfYes) / len(tweets)
    features['q1']['no']['probability'] = float(numberOfNo) / len(tweets)



#----MAIN----
openFileTrainingSet()

getVocabulary()
getProbabiltyOfQ1()
getTotalAndFrequencyOfWordsOfQ1()
getProbabilityForEachWordQ1()

print()