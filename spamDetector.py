import csv
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from math import log

def readTrainTestData():
    with open('data/train_test.csv', 'r') as f:
        reader = csv.reader(f)
        trainTestData = list(reader)
    return trainTestData[1:]

class SpamDetector:
    def __init__(self, trainData, testData):
        self.trainData = trainData
        self.testData = testData 

    def normalizeText(self, text):
        stopWords = set(stopwords.words('english'))
        punctuation = string.punctuation
        stemmer = PorterStemmer()
        
        text = text.translate(str.maketrans('', '', punctuation))
        wordTokens = word_tokenize(text)
        filteredWords = [w.lower() for w in wordTokens if not w in stopWords]
        stemmedWords = [stemmer.stem(w) for w in filteredWords]
        return stemmedWords

    def train(self):
        self.numSpamWords, self.numSpamEmails, self.numHamWords, self.numHamEmails, self.numSpamDigit, self.numHamDigit = 0, 0, 0, 0, 0, 0
        self.spamWords = dict()
        self.hamWords = dict()
        
        for item in self.trainData:
            label = item[0]
            text = item[1]
            
            if label == 'ham':
                self.numHamEmails += 1
            else:
                self.numSpamEmails += 1
            
            words = self.normalizeText(text)
            for word in words:
                if label == 'ham':
                    self.numHamWords += 1
                    if word.isdigit():
                        self.numHamDigit += 1
                    else:
                        self.hamWords[word] = self.hamWords.get(word, 0) + 1     
                else:
                    self.numSpamWords +=1
                    if word.isdigit():
                        self.numSpamDigit += 1
                    else:
                        self.spamWords[word] = self.spamWords.get(word, 0) + 1                
        self.calcProb()


    def calcProb(self):
        self.hamWordsProb = dict()
        self.spamWordsProb = dict()
        numEmails = self.numHamEmails + self.numSpamEmails
        self.hamEmailProb = self.numHamEmails / numEmails
        self.spamEmailProb = self.numSpamEmails / numEmails 
        self.numDistinctHamWords = len(list(self.hamWords.keys()))
        self.numDistinctSpamWords = len(list(self.spamWords.keys()))

        for word in self.spamWords.keys():
            self.spamWordsProb[word] = ((self.spamWords[word] + 1) / (self.numSpamWords + self.numDistinctSpamWords))    
        for word in self.hamWords.keys():
            self.hamWordsProb[word] = ((self.hamWords[word] + 1) / (self.numHamWords + self.numDistinctHamWords))
       
        self.hamDigitProb = self.numHamDigit / self.numHamWords
        self.hamLetterProb = 1 - self.hamDigitProb
        self.spamDigitProb = self.numSpamDigit / self.numSpamWords
        self.spamLetterProb = 1 - self.spamDigitProb      
        # s = max(self.hamWordsProb, key=self.hamWordsProb.get)
        # print(s, self.hamWordsProb[s])

    def isSpam(self, text):
        spamProb = log(self.spamEmailProb)
        hamProb = log(self.hamEmailProb)
        words = self.normalizeText(text)
       
        for word in words:
            if word.isdigit():
                spamProb += log(self.spamDigitProb)
                hamProb += log(self.hamDigitProb)
            else:
                spamProb += log(self.spamLetterProb)
                hamProb += log(self.hamLetterProb)
                if word in self.spamWordsProb:
                    spamProb += log(self.spamWordsProb[word])
                else:
                    spamProb += log( 1 / (self.numSpamWords + self.numDistinctSpamWords) )
            
                if word in self.hamWordsProb:
                    hamProb += log(self.hamWordsProb[word])
                else:
                    hamProb += log( 1 / (self.numHamWords + self.numDistinctHamWords) )
        return spamProb >= hamProb
    
    def predict(self, testTexts):
        result = dict()
        for i, text in enumerate(testTexts):
            if self.isSpam(text):
                result[i] = 'spam'
            else:
                result[i] = 'ham'
        return result

    def test(self):
        testTexts = [item[1] for item in self.testData]
        testLabels = [item[0] for item in self.testData]
        truePositive, trueNegative, falsePositive, falseNegative = 0, 0, 0, 0

        result = self.predict(testTexts)
        for i, label in enumerate(testLabels):
            if result[i] == 'spam' and label == 'spam':
                truePositive += 1
            elif result[i] == 'ham' and label == 'ham':
                trueNegative += 1
            elif result[i] == 'spam' and label == 'ham':
                falsePositive += 1
            elif result[i] == 'ham' and label == 'spam':
                falseNegative += 1

        self.estimateTestResult(truePositive, trueNegative, falsePositive, falseNegative)

    def estimateTestResult(self, truePositive, trueNegative, falsePositive, falseNegative):
        recall = (truePositive / (truePositive + falseNegative)) * 100
        precision = (truePositive / (truePositive + falsePositive)) * 100
        accuracy = ((trueNegative + truePositive) / (trueNegative + truePositive + falseNegative + falsePositive)) * 100
        
        print('Recall', recall, '%')
        print('Precision', precision, '%')
        print('Accuracy', accuracy, '%')

data = readTrainTestData()
trainPortion = int(0.75 * len(data))
trainData = data[0:trainPortion]
testData = data[trainPortion+1 :]

detector = SpamDetector(trainData, trainData)
detector.train()
detector.test()
