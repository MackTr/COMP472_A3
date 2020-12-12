from collections import Counter
import numpy as np
trainingData = "covid_training.tsv"
testData = "covid_test_public.tsv"
lolol = "test.tsv"

with open(lolol) as f:
    next(f)
    tweet = []
    label = []
    for line in f:
        #split line into list of strings and lowercase
        data = line.lower().split()
        #remove first (tweetID) and last 7 (trucs wack) indexes
        label.append(data[-7:-6])
        tweet.append(data[1:-7])


#Probabilité de chaque classes (  YES / NO)
label = np.array(label)
occ = label.reshape(1,len(label))
NbYes = (occ == 'yes').sum()
NbNo = (occ == 'no').sum()

ProbabilityYes = NbYes / (NbNo + NbYes)
ProbabilityNo = NbNo / (NbNo + NbYes)


# Occurence of each word by class (class = yes/no)
vocabularyYes = Counter()
vocabularyNo = Counter()
for i in range(len(tweet)):

    if label[i] == 'yes':
        for word in tweet[i]:
            vocabularyYes[word] += 1

    if label[i] == 'no':
        for word in tweet[i]:
            vocabularyNo[word] += 1

totalWordsYes = sum(Counter(vocabularyYes).values())
totalWordsNo = sum(Counter(vocabularyNo).values())

#get probabilities of each word by class
for word in vocabularyYes:
   vocabularyYes[word] /= totalWordsYes

for word in vocabularyNo:
   vocabularyNo[word] /= totalWordsNo

print(vocabularyNo)