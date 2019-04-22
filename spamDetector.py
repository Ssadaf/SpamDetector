import csv
# import pandas as pd
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

def readTrainTestData():
    with open('data/train_test.csv', 'r') as f:
        reader = csv.reader(f)
        trainTestData = list(reader)
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

class SpamDetector:
    def __init__(self, trainData, testData):
        self.trainData = trainData
        self.testData = testData 

    def train(self):
        self.numSpamWords = 0
        self.numHamWords = 0
        self.spamWords = dict()
        self.hamWords = dict()
        
        for item in self.trainData:
            label = item[0]
            text = item[1]
            words = normalizeText(text)
            for word in words:
                if label == 'ham':
                    self.numHamWords += 1
                    self.hamWords[word] = self.hamWords.get(word, 0) + 1
                else:
                    self.numSpamWords +=1
                    self.spamWords[word] = self.spamWords.get(word, 0) + 1
        self.calcProb()


    def calcProb(self):
        # self.probHam = self.numHamWords/
        # self.probSpam = 0
        self.hamWordsProb = dict()
        self.spamWordsProb = dict()
        self.numDistinctHamWords = len(list(self.hamWords.keys()))
        self.numDistinctSpamWords = len(list(self.spamWords.keys()))

        for word in self.spamWords.keys():
            self.spamWordsProb[word] = ((self.spamWords[word] + 1) / (self.numSpamWords + self.numDistinctSpamWords))    
        for word in self.hamWords.keys():
            self.hamWordsProb[word] = ((self.hamWords[word] + 1) / (self.numHamWords + self.numDistinctHamWords))

data = readTrainTestData()
trainPortion = int(0.75 * len(data))
trainData = data[0:trainPortion]
testData = data[trainPortion+1 :]

detector = SpamDetector(trainData, testData)
detector.train()