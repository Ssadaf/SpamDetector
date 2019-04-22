import csv
# import pandas as pd
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from collections import Counter

def readTrainTestData():
    with open('data/train_test.csv', 'r') as f:
        reader = csv.reader(f)
        trainTestData = list(reader)
    # for item in trainTestData:
    #     print(item)
    return trainTestData[1:]

def normalizeText(text):
    stopWords = set(stopwords.words('english'))
    punctuation = string.punctuation
    stemmer = PorterStemmer()
    
    text = text.translate(str.maketrans('', '', punctuation))
    wordTokens = word_tokenize(text)
    filteredWords = [w.lower() for w in wordTokens if not w in stopWords]
    stemmedWords = [stemmer.stem(w) for w in filteredWords]
    return stemmedWords

def refineTextInTrainingData(data):
    refined = []
    for item in data:
        text = normalizeText(item[1])
        refined.append([item[0], text])    
    return refined

data = readTrainTestData()
trainPortion = int(0.75 * len(data))
trainData = data[0:trainPortion]
testData = data[trainPortion+1 :]
refTrainData = refineTextInTrainingData(trainData)

for item in refTrainData[0:6]:
    print(item)
