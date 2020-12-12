import math
trainingData = "covid_training.tsv"
testData = "covid_test_public.tsv"

test_tweets = []
training_tweets = []
vocabulary = {}
features = {'q1': {'yes': {'totalWords': 0, 'probability': 0}, 'no': {'totalWords': 0, 'probability': 0}}}

results = []

smoothingValue = 0.01

def openFileTrainingSet():
    with open(trainingData, encoding="utf8") as file:
        next(file)

        for line in file:
            # split line into list of strings and lowercase
            data = line.lower().split()
            # remove first (tweetID) and last 6 indexes
            label = data[-7:-6]
            content = data[1:-7]
            training_tweets.append({"content": content, "label" : label})

def openFileTestSet():
    with open(testData, encoding="utf8") as file:

        for line in file:
            # split line into list of strings and lowercase
            data = line.lower().split()
            # remove first (tweetID) and last 6 indexes
            label = data[-7:-6]
            content = data[1:-7]
            test_tweets.append({"content": content, "label": label})

def getVocabulary():

    for tweet in training_tweets:
        for word in tweet['content']:
            if word not in vocabulary:
                vocabulary[word] = {'frequency_q1_yes': 0, 'frequency_q1_no': 0, 'p_q1_yes': 0, 'p_q1_no': 0}

def getTotalAndFrequencyOfWordsOfQ1():

    for tweet in training_tweets:
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

    for tweet in training_tweets:
        if tweet['label'] == ['yes']:
            numberOfYes += 1
        else:
            numberOfNo += 1

    features['q1']['yes']['probability'] = float(numberOfYes) / len(training_tweets)
    features['q1']['no']['probability'] = float(numberOfNo) / len(training_tweets)

def getScoresForAllWords(words, label):
    score = 0

    for word in words:
        if word in vocabulary:
            score += math.log10(vocabulary[word]['p_q1_' + label])
        else:
            score += math.log10(smoothingValue / (features['q1'][label]['totalWords'] + smoothingValue))

    return score

def testModel():
    for tweet in test_tweets:
        score_q1_yes = math.log10(features['q1']['yes']['probability']) + getScoresForAllWords(tweet['content'], 'yes')
        score_q1_no = math.log10(features['q1']['no']['probability']) + getScoresForAllWords(tweet['content'], 'no')

        if score_q1_yes > score_q1_no:
            results.append('yes')
        else:
            results.append('no')

    for index in range(0, len(results)):
        if results[index] == test_tweets[index]['label'][0]:
            print('true')
        else:
            print('false')


def trainingPhase():
    openFileTrainingSet()
    getVocabulary()
    getProbabiltyOfQ1()
    getTotalAndFrequencyOfWordsOfQ1()
    getProbabilityForEachWordQ1()

def testingPhase():
    openFileTestSet()
    testModel()


#----MAIN----
trainingPhase()
testingPhase()

print()