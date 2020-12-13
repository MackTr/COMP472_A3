import math
trainingData = "covid_training.tsv"
testData = "covid_test_public.tsv"

test_tweets = []
training_tweets = []
vocabulary = {}
features = {'q1': {'yes': {'totalWords': 0, 'probability': 0}, 'no': {'totalWords': 0, 'probability': 0}}}
name_trace_file = "txt/trace_NB-BOW-OV.txt"
name_eval_file = "txt/eval_NB-BOW-OV.txt"

results = []

smoothingValue = 0.01

# Open training set file and adds it to the training_tweets array
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

# Open training set file and adds it to the test_tweets array
def openFileTestSet():
    with open(testData, encoding="utf8") as file:

        for line in file:
            # split line into list of strings and lowercase
            data = line.lower().split()
            # remove first (tweetID) and last 6 indexes
            label = data[-7:-6]
            content = data[1:-7]
            id = data[0]
            test_tweets.append({"content": content, "label": label, "id": id})

# Creates the vocabulary dictionary
def getVocabulary():

    for tweet in training_tweets:
        for word in tweet['content']:
            if word not in vocabulary:
                vocabulary[word] = {'frequency_q1_yes': 0, 'frequency_q1_no': 0, 'p_q1_yes': 0, 'p_q1_no': 0}

# Add the frequency of each word of the vocabulary and counts the total words for each class
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

# Compute the P(WORD/CLASS) for each word of the vocabulary with smoothing
def getProbabilityForEachWordQ1():

    smoothingOfDenominator = smoothingValue * len(vocabulary)

    for word in vocabulary:
        vocabulary[word]['p_q1_yes'] = (vocabulary[word]['frequency_q1_yes'] + smoothingValue) / (features['q1']['yes']['totalWords'] + smoothingOfDenominator)
        vocabulary[word]['p_q1_no'] = (vocabulary[word]['frequency_q1_no'] + smoothingValue) / (features['q1']['no']['totalWords'] + smoothingOfDenominator)

#Compute the P(CLASS)
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

# Compute the score of the given tweet for Q1
def getScoresForAllWords(words, label):
    score = 0
    smoothingOfDenominator = smoothingValue * len(vocabulary)

    for word in words:
        if word in vocabulary:
            score += math.log10(vocabulary[word]['p_q1_' + label])
        else:
            score += math.log10(smoothingValue / (features['q1'][label]['totalWords'] + smoothingOfDenominator))

    return score

# Run compute the score of each tweets and find the result
def testModel():
    for tweet in test_tweets:
        score_q1_yes = math.log10(features['q1']['yes']['probability']) + getScoresForAllWords(tweet['content'], 'yes')
        score_q1_no = math.log10(features['q1']['no']['probability']) + getScoresForAllWords(tweet['content'], 'no')

        if score_q1_yes > score_q1_no:
            results.append({'class': 'yes', 'score_yes': score_q1_yes, 'score_no': score_q1_no})
        else:
            results.append({'class': 'no', 'score_yes': score_q1_yes, 'score_no': score_q1_no})

# Return scientific notation of given number
def sciNotation(number):
    return "{:.2e}".format(number)

# Print trace file
def printTraceFile():
    file = open(name_trace_file, 'w')
    for index in range (0, len(results)):
        line = test_tweets[index]['id'] + "  "
        line += results[index]['class'] + "  "
        line += sciNotation(results[index]['score_' + results[index]['class']]) + "  "
        line +=  test_tweets[index]['label'][0] + "  "
        if results[index]['class'] == test_tweets[index]['label'][0]:
            line += "correct\n"
        else:
            line += "wrong\n"

        file.write(line)

# Compute accuracy of test
def getAccuracy():
    total = len(test_tweets)
    correct_values = 0
    for index in range(0, len(test_tweets)):
        if results[index]['class'] == test_tweets[index]['label'][0]:
            correct_values += 1
    return str(float(correct_values)/total) + '\n'

# Compute precision of test
def getPrecision():
    total_yes = 0
    true_yes = 0

    total_no = 0
    true_no = 0

    for index in range(0, len(test_tweets)):
        if results[index]['class'] == "yes":
            total_yes += 1
            if results[index]['class'] == test_tweets[index]['label'][0]:
                true_yes += 1
        else:
            total_no +=1
            if results[index]['class'] == test_tweets[index]['label'][0]:
                true_no += 1

    return float(true_yes)/total_yes, float(true_no)/total_no

# Compute recall of test
def getRecall():
    total_yes = 0
    true_yes = 0

    total_no = 0
    true_no = 0

    for index in range(0, len(test_tweets)):
        if results[index]['class'] == "yes":
            if results[index]['class'] == test_tweets[index]['label'][0]:
                true_yes += 1
        else:
            if results[index]['class'] == test_tweets[index]['label'][0]:
                true_no += 1

    for tweet in test_tweets:
        if tweet['label'] == ['yes']:
            total_yes += 1
        else:
            total_no += 1

    return float(true_yes)/total_yes, float(true_no)/total_no

# Print eval file
def printEvalFile():
    file = open(name_eval_file, 'w')
    file.write(getAccuracy())
    precision_yes, precision_no = getPrecision()
    file.write(str(precision_yes) + "  " + str(precision_no) + "\n")
    recall_yes, recall_no = getRecall()
    file.write(str(recall_yes) + "  " + str(recall_no) + "\n")
    f1_yes = ((1 + 1) * precision_yes * recall_yes) / (precision_yes + recall_yes)
    f1_no = ((1 + 1) * precision_no * recall_no) / (precision_no + recall_no)
    file.write(str(f1_yes) + "  " + str(f1_no) + "\n")

# training steps
def trainingPhase():
    openFileTrainingSet()
    getVocabulary()
    getProbabiltyOfQ1()
    getTotalAndFrequencyOfWordsOfQ1()
    getProbabilityForEachWordQ1()

# testing steps
def testingPhase():
    openFileTestSet()
    testModel()
    printTraceFile()
    printEvalFile()

#----MAIN----
trainingPhase()
testingPhase()

print()