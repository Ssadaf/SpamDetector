import csv
import string
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from math import log


def readData(path):
    with open(path, 'r') as f:
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
        
        text = text.translate(str.maketrans(punctuation, '                                '))
        wordTokens = word_tokenize(text)
        wordTokens = [w.lower() for w in wordTokens]
        filteredWords = [w for w in wordTokens if not w in stopWords]
        stemmedWords = [stemmer.stem(w) for w in filteredWords]
        return stemmedWords

    def check(self, text):
        stopWords = set(stopwords.words('english'))
        punctuation = string.punctuation
        stemmer = PorterStemmer()
        print("________________________")
        print(text)
        text = text.translate(str.maketrans(punctuation, '                                '))        
        # text = text.translate(str.maketrans('', '', punctuation))
        print("PUNC: ", text)
        wordTokens = word_tokenize(text)
        wordTokens = [w.lower() for w in wordTokens]
        filteredWords = [w for w in wordTokens if not w in stopWords]
        print("FILTER: ", filteredWords)
        stemmedWords = [stemmer.stem(w) for w in filteredWords]
        print("STEM: ", stemmedWords)
        print("_________________________")
        return stemmedWords

    def giveSizeInterval(self, size):
        unit = self.intervalUnit
        if size < unit:
            return 1
        elif size < 2 * unit:
            return 2
        elif size < 3 * unit:
            return 3
        elif size < 4 * unit:
            return 4
        else:
            return 5

    def putEmailSizesInInterval(self, sizes):
        result = dict()
        interval, count = 1, 0
        unit = self.intervalUnit

        for size in sizes:
            if size < interval * unit:
                count += 1
            else:
                while interval < self.giveSizeInterval(size):
                    result[interval] = count
                    interval += 1
                    count = 0
        
        while interval <= 5:
            result[interval] = 0
            interval += 1
        
        return result

    def handleEmailSizes(self, hamSizes, spamSizes):
        hamSizes.sort()
        spamSizes.sort()

        minSize = min(hamSizes[0], spamSizes[0])
        maxSize = max(hamSizes[len(hamSizes)-1], spamSizes[len(spamSizes)-1])
        self.intervalUnit = int((maxSize - minSize) / 4)
        
        self.hamEmailSizes = self.putEmailSizesInInterval(hamSizes)
        self.spamEmailSizes = self.putEmailSizesInInterval(spamSizes)
    
    def train(self):
        self.numSpamWords, self.numSpamEmails, self.numHamWords, self.numHamEmails = 0, 0, 0, 0 
        self.numSpamDigit, self.numHamDigit, self.numSpamPhone, self.numHamPhone = 0, 0, 0, 0
        self.spamWords, self.hamWords = dict(), dict()
        hamEmailSizes, spamEmailSizes  = [], []
        
        for item in self.trainData:
            label = item[0]
            text = item[1]
            words = self.normalizeText(text)
            for word in words:
                if label == 'ham':
                    self.numHamWords += 1
                    if word.isdigit():
                        # if word.startswith('09'):
                        #     self.numHamPhone += 1
                        # else:
                        self.numHamDigit += 1
                    else:
                        self.hamWords[word] = self.hamWords.get(word, 0) + 1     
                else:
                    self.numSpamWords +=1
                    if word.isdigit():
                        # if word.startswith('09'):
                        #     self.numSpamPhone += 1
                        # else:
                        self.numSpamDigit += 1
                    else:
                        self.spamWords[word] = self.spamWords.get(word, 0) + 1               
                             
            if label == 'ham':
                self.numHamEmails += 1
                hamEmailSizes.append(len(text))
            else:
                self.numSpamEmails += 1 
                spamEmailSizes.append(len(text))
        self.drawSizePlot(spamEmailSizes, hamEmailSizes)
        #self.drawWords()
        #self.numberOfDigits()
        self.handleEmailSizes(hamEmailSizes, spamEmailSizes)
        self.calcProb()

    def calcProb(self):
        self.hamWordsProb, self.spamWordsProb, self.hamSizeProb, self.spamSizeProb = dict(), dict(), dict(), dict()
        numEmails = self.numHamEmails + self.numSpamEmails
        self.hamEmailProb = self.numHamEmails / numEmails
        self.spamEmailProb = self.numSpamEmails / numEmails 
        self.numDistinctHamWords = len(list(self.hamWords.keys()))
        self.numDistinctSpamWords = len(list(self.spamWords.keys()))

        for word in self.spamWords.keys():
            self.spamWordsProb[word] = ((self.spamWords[word] + 1) / (self.numSpamWords + self.numDistinctSpamWords))    
        for word in self.hamWords.keys():
            self.hamWordsProb[word] = ((self.hamWords[word] + 1) / (self.numHamWords + self.numDistinctHamWords))
        for size in self.spamEmailSizes.keys():
            self.spamSizeProb[size] = ((self.spamEmailSizes[size] + 1) / (self.numSpamEmails + len(list(self.spamEmailSizes.keys()))))
        for size in self.hamEmailSizes.keys():
            self.hamSizeProb[size] = ((self.hamEmailSizes[size] + 1) / (self.numHamEmails + len(list(self.hamEmailSizes.keys()))))
       
        self.hamDigitProb = self.numHamDigit / self.numHamWords
        # self.hamPhoneProb = ((self.numHamPhone + 1) / (self.numHamWords + 2))
        self.hamLetterProb = 1 - self.hamDigitProb
        self.spamDigitProb = self.numSpamDigit / self.numSpamWords
        # self.spamPhoneProb = ((self.numSpamPhone + 1) / (self.numSpamWords + 2))
        self.spamLetterProb = 1 - self.spamDigitProb 
 
    def makeWordBagSmaller(self, wordBag):
        l = wordBag.items()
        lsorted = sorted(l, key = lambda l:l[1], reverse=True)
        newWordBag = lsorted[0:20]
        return newWordBag

    def drawWords(self):
        spamTopTen = self.makeWordBagSmaller(self.spamWords)
        keys = [item[0] for item in spamTopTen]
        values = [item[1] for item in spamTopTen]
        plt.barh(keys, values)
        plt.xlabel("Word")
        plt.ylabel("Frequency")
        plt.title("Top 20 Frequent Words in Spams")

        plt.show() 

        hamTopTen = self.makeWordBagSmaller(self.hamWords)
        keys = [item[0] for item in hamTopTen]
        values = [item[1] for item in hamTopTen]
        plt.barh(keys, values)
        plt.xlabel("Word")
        plt.ylabel("Frequency")
        plt.title("Top 20 Frequent Words in Hams")

        plt.show() 

    def numberOfDigits(self):
        keys = ["spam", "ham"]
        values = [self.numSpamDigit, self.numHamDigit]
        plt.bar(keys, values)
        plt.xlabel("class")
        plt.ylabel("Frequency")
        plt.title("Occurance of Numbers")

        plt.show()

    def drawSizePlot(self, spamEmailSizes, hamEmailSizes):
        plt.subplot(211)
        plt.hist(spamEmailSizes,color = 'blue')
        plt.xlabel("Size of Email")
        plt.ylabel("Frequency")
        plt.title("Size of Spam Emails")
        
        plt.subplot(212)
        plt.hist(hamEmailSizes, color = 'orange')
        plt.xlabel("Size of Email")
        plt.ylabel("Frequency")
        plt.title("Size of Ham Emails")
        
        plt.show() 


    def isSpam(self, text):
        spamProb = log(self.spamEmailProb)
        hamProb = log(self.hamEmailProb)
        words = self.normalizeText(text)   
        for word in words:
            if word.isdigit():
                # if word.startswith('09'):
                #     spamProb += log(self.spamPhoneProb)
                #     hamProb += log(self.hamPhoneProb)
                # else:
                spamProb += log(self.spamDigitProb)
                hamProb += log(self.hamDigitProb)
            else:
                spamProb += log(self.spamLetterProb)
                hamProb += log(self.hamLetterProb)
                if word in self.spamWordsProb:
                    spamProb += log(self.spamWordsProb[word])
                else:
                    spamProb += log( 1 / (self.numSpamWords + self.numDistinctSpamWords) )
                    # spamProb -= 10
                if word in self.hamWordsProb:
                    hamProb += log(self.hamWordsProb[word])
                else:
                    hamProb += log( 1 / (self.numHamWords + self.numDistinctHamWords) )
                    # hamProb -= 10

        spamProb += log(self.spamSizeProb[self.giveSizeInterval(len(text))])
        hamProb += log(self.hamSizeProb[self.giveSizeInterval(len(text))])
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
                # self.check(testTexts[i])                                
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
    
    def evaluate(self, data, outputFile):
        texts = [item[1] for item in data]
        ids = [item[0] for item in data]
    
        prediction = self.predict(texts)
        results = [['id','type']]
        for i in range(len(ids)):
            results.append([ids[i], prediction[i]])
        
        with open(outputFile, 'w') as writeFile:
            writer = csv.writer(writeFile)
            writer.writerows(results)
        writeFile.close()

trainTestData = readData('data/train_test.csv')
trainPortion = int(0.8 * len(trainTestData))
trainData = trainTestData[0:trainPortion]
testData = trainTestData[trainPortion+1 :]

detector = SpamDetector(trainData, testData)
detector.train()
detector.test()

evaluateData = readData('data/evaluate.csv')
detector.evaluate(evaluateData, 'output.csv')