from collections import Counter


trainingData = "covid_training.tsv"
testData = "covid_test_public.tsv"
lolol = "test.tsv"

with open(trainingData) as f:
    vocabulary = Counter()
    next(f)
    for line in f:
        #split line into list of strings and lowercase
        tweet = line.lower().split()
        #remove first (tweetID) and last 7 (trucs wack) indexes
        tweet = tweet[1:-7]
        for word in tweet:

            if word in vocabulary:
                vocabulary[word] += 1
            else:
                vocabulary[word] += 1            

## print full vocabulary including word with occurance of 1
#print(vocabulary)

##remove all occurences of 1 in vocabulary
filteredVoc =  Counter(el for el in vocabulary.elements() if vocabulary[el] > 1)

#print updated vocab
#print(filteredVoc)
print(sum(Counter(filteredVoc).values()))

totalWords = sum(Counter(filteredVoc).values())

for word in filteredVoc:
    filteredVoc[word] /= totalWords

print(filteredVoc)


