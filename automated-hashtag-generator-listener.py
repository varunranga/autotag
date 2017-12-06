from __future__ import division, print_function, absolute_import, unicode_literals

import nltk
import math
import re
import csv
import pandas as pd
import pickle
import sys
import time
import threading
import string
import keras
import random
import json
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Activation
from nltk import word_tokenize
from nltk.tokenize import TweetTokenizer
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from collections import Counter
from nltk.corpus import stopwords
from string import punctuation
from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
from textblob import TextBlob as tb
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.svm import SVC, LinearSVC

ckey = 'xTgwF2Kft3Rgo5FfQevFowj57'
csecret = '4n7PAMIF1pTjXpzAaVTw2KyxOBPbNLPh51aj9kT55qmdt9vFrB'
atoken = '796775752054120448-W6y3Taoo4HpeZIewukPlLVUifmVJ4Iv'
asecret = 'jiCu2TbFQspV0Oaj4dnOhT86D90cIsvEWREZVwwjvCusv'

THRESHOLD_NB  = 10
THRESHOLD_SVM = 10000
THRESHOLD_DNN = 500000

datasetBuffer_NB = []
datasetBuffer_SVM = []
datasetBuffer_DNN = []
datasetBuffer_DNN_InputVectors = []

bagOfWords = [word.strip() for word in open('Glove_BagOfWords.txt', 'r').readlines()]

abbreviationsDict = {}
WORDS = None

class listener(StreamListener):
	def on_data(self, data):
		try:
			jsonObj = json.loads(bytes(data, "utf-8").decode('utf-8'))

			try:
				tweet = jsonObj['text']
			except:
				tweet = ""

			print("Tweepy Listener:", tweet)

			cleanedString, extractedHashtag = CleanStringAndExtractHashtags(tweet)
			
			threading.Thread(target = AddToBuffers, kwargs = {'cleanedString': cleanedString, 'extractedHashtags': extractedHashtag}).start()
			
			return True
		except:
			print('Tweepy Listener: Failed, but calm down')
			time.sleep(1)
			
	def on_error(self, status):
		print ("status",status)

# Functions for tf-idf

def tf(word, blob):
    return blob.words.count(word) / len(blob.words)

def n_containing(word, bloblist):
    return sum(1 for blob in bloblist if word in blob.words)

def idf(word, bloblist):
    return math.log(len(bloblist) / (1 + n_containing(word, bloblist)))

def tfidf(word, blob, bloblist):
    return tf(word, blob) * idf(word, bloblist)


tweetsReceived = 0
def AddToBuffers(cleanedString, extractedHashtags):
	global tweetsReceived

	def convertSentenceToVector(string):
		print("AddToBuffers: Creating Sentence to Vector")

		wordList = word_tokenize(string)
		vector = [0 for _ in range(len(bagOfWords))]
		for word in wordList:
			try:
				vector[bagOfWords.index(word)] += 1
			except:
				pass
		return vector

	datasetBuffer_NB.append([cleanedString, extractedHashtags])
	datasetBuffer_SVM.append([cleanedString, extractedHashtags])

	datasetBuffer_DNN.append([cleanedString, extractedHashtags])
	datasetBuffer_DNN_InputVectors.append(convertSentenceToVector(cleanedString))

	tweetsReceived += 1

	print(tweetsReceived, ": Added to buffers")

	if ((tweetsReceived % THRESHOLD_DNN) == 0):
		threading.Thread(target = ThresholdDNNReached).start()
	elif((tweetsReceived % THRESHOLD_SVM) == 0):
		threading.Thread(target = ThresholdSVMReached).start()
	elif((tweetsReceived % THRESHOLD_NB) == 0):
		threading.Thread(target = ThresholdNBReached).start()

def ThresholdDNNReached():
	print("Threshold DNN Reached")
	DNN_Train()
	SVM_BufferClear()
	NB_BufferClear()

def ThresholdSVMReached():
	print("Threshold SVM Reached")
	SVM_Train()
	NB_BufferClear()

def ThresholdNBReached():
	print("Threshold NB Reached")
	NB_Train()

def SVM_BufferClear():
	print("SVM Buffer Cleared")
	del datasetBuffer_SVM[:THRESHOLD_DNN]

def NB_BufferClear():
	print("NB Buffer Cleared")
	del datasetBuffer_SVM[:THRESHOLD_SVM]

def DNN_Train():
	print("DNN_Train: Started.")

	cleanedStrings = []
	extractedHashtags = []

	for cleanedString, extractedHashtag in datasetBuffer_DNN:
		cleanedStrings.append(cleanedString)
		extractedHashtags.append(extractedHashtag)

	print("DNN_Train: Creating bloblist for TF-IDF.")
	bloblist = [tb(text) for text in cleanedStrings]

	def CreateBagOfHashtags():

		print("DNN_Train: Creating bag of hashtags.")

		def RemovePunctuations(word):
			string = "".join(character for character in string.strip() if character not in punctuation)

			return string

		hashtags = []
		scores = []

		j = len(extractedHashtags)

		for i in range(j):
			if (len(extractedHashtags[i]) > 0):
				for hashtag in extractedHashtags[i]:
					hashtags.append(hashtag[1:])
					scores.append(1)
			else:
				scoresTB = {word: tfidf(word, bloblist[i], bloblist) for word in bloblist[i].words}
				sorted_words = sorted(scoresTB.items(), key=lambda x: x[1], reverse=True)
				try:
					word, score = sorted_words[0]
					hashtags.append(word)
					scores.append(score)
				except:
					hashtags.append('')
					scores.append(0)

		return hashtags, scores

	hashtags, scores = CreateBagOfHashtags()
	bagOfHashtags = sorted(list(set(hashtags)))

	print("DNN_Train: Saving bag of hashtags.")

	bagOfHashtagsFile = open('bagOfHashtags.pickle', 'wb')
	pickle.dump(bagOfHashtags, bagOfHashtagsFile)
	bagOfHashtags.close()

	vectors = []

	j = len(extractedHashtags)
	i = 0

	print("DNN_Train: Creating class vectors, one-hot encoded.")

	for k in range(j):
		currentClass = [0 for _ in range(len(bagOfHashtags))]
		if (len(extractedHashtags[k]) > 0):
			for hashtag in extractedHashtags[k]:
				currentClass[bagOfHashtags.index(hashtag[1:])] = 1
				i += 1
		else:
			currentClass[bagOfHashtags.index(hashtags[i])] = scores[i]
			i += 1
		vectors.append(currentClass)

	trainX = datasetBuffer_DNN_InputVectors
	trainY = vectors

	print("DNN_Train: Creating training model.")

	model = Sequential()
	model.add(Dense(units = 512, activation = 'sigmoid', input_dim = len(trainX[0])))
	model.add(Dense(units = 768, activation = 'sigmoid'))
	model.add(Dense(units = 1024, activation = 'sigmoid'))
	model.add(Dense(units = len(trainY[0])))
	model.add(Activation('softmax'))
	model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.SGD(lr = 0.01, momentum = 0.9, nesterov = True))
	model.fit(trainX, trainY, validation_data = (trainX, trainY), epochs = 100, batch_size = 64)

	print("DNN_Train: Saving model.")

	model.save('Keras_Train_ANN.model')	

	print("DNN_Train: Done.")

def SVM_Train():
	print("SVM_Train: Starting.")

	wordsAllInOne = {}
	hashtagAllInOne = {}
	dct = {}

	tweets = []
	hashtags = []

	for cleanedString, extractedHashtag in datasetBuffer_NB:
		tweets.append(cleanedString)
		hashtags.append(extractedHashtag)

	def MakeDictionary(tweetsF, hashtagsF):
		nonlocal wordsAllInOne
		nonlocal hashtagAllInOne
		tweets = tweetsF
		hashtags = hashtagsF		
		wordsAllInOne = set([word for tweet in tweets for word in word_tokenize(tweet)])
		hashtagAllInOne = set([hashtag for hashtagList in hashtags for hashtag in hashtagList])
		
		dct = {hashtag:[] for hashtag in hashtagAllInOne}
		for hashtag in hashtagAllInOne:
			indexes = [i for i in range (len(hashtags)) if hashtag in hashtags[i]]
			for index in indexes:
				filtered_sentence = []
				words = word_tokenize(tweets[index])
				for w in words:
					filtered_sentence.append(w)
				dct[hashtag] += filtered_sentence
		return dct 

	print("SVM_Train: Making dictionary for NLTK SVM Classifier.")
	dct = MakeDictionary(tweets, hashtags)

	documents = [
					(wordList, hashtag) 
					for hashtag in hashtagAllInOne 
					for wordList in dct[hashtag]
				]

	 
	random.shuffle(documents)

	all_words = []

	for w in wordsAllInOne:
		all_words.append(w.lower())

	all_words = nltk.FreqDist(all_words)
	word_features = list(all_words.keys())

	fileObj = open('WordFeaturesNB.pickle', 'wb')
	pickle.dump(word_features, fileObj)
	fileObj.close()

	def findFeatures(document):
		wordsD = set(document)
		features = {w: (w in wordsD) for w in word_features}
		return features

	print("SVM_Train: Finding features.")
	featureSet = [(findFeatures(rev), category) for (rev, category) in documents]

	print("SVM_Train: Creating LinearSVC Classifier. Training.")
	classifier = SklearnClassifier(LinearSVC())
	classifier.train(featureSet)


	print("SVM_Train: Accuracy percent :", (nltk.classify.accuracy(classifier, featureSet))*100)

	print("SVM_Train: Saving classifier.")
	fileObj = open('SVM_Classifier.pickle', 'wb')
	pickle.dump(classifier, fileObj)
	fileObj.close()

	print("SVM_Train: Done.")


def NB_Train():
	print("NB_Train: Starting.")

	wordsAllInOne = {}
	hashtagAllInOne = {}
	dct = {}

	tweets = []
	hashtags = []

	for cleanedString, extractedHashtag in datasetBuffer_NB:
			tweets.append(cleanedString)
			hashtags.append(extractedHashtag)

	def MakeDictionary(tweetsF, hashtagsF):
		nonlocal wordsAllInOne
		nonlocal hashtagAllInOne
		tweets = tweetsF
		hashtags = hashtagsF		
		wordsAllInOne = set([word for tweet in tweets for word in word_tokenize(tweet)])
		hashtagAllInOne = set([hashtag for hashtagList in hashtags for hashtag in hashtagList])
		
		dct = {hashtag:[] for hashtag in hashtagAllInOne}
		for hashtag in hashtagAllInOne:
			indexes = [i for i in range (len(hashtags)) if hashtag in hashtags[i]]
			for index in indexes:
				filtered_sentence = []
				words = word_tokenize(tweets[index])
				for w in words:
					filtered_sentence.append(w)
				dct[hashtag] += filtered_sentence
		return dct 

	print("NB_Train: Making dictionary for NLTK SVM Classifier.")
	dct = MakeDictionary(tweets, hashtags)

	documents = [
					(wordList, hashtag) 
					for hashtag in hashtagAllInOne 
					for wordList in dct[hashtag]
				]

	 
	random.shuffle(documents)

	all_words = []

	for w in wordsAllInOne:
		all_words.append(w.lower())

	all_words = nltk.FreqDist(all_words)
	word_features = list(all_words.keys())

	fileObj = open('WordFeaturesNB.pickle', 'wb')
	pickle.dump(word_features, fileObj)
	fileObj.close()

	def findFeatures(document):
		wordsD = set(document)
		features = {w: (w in wordsD) for w in word_features}
		return features

	print("NB_Train: Finding features.")
	featureSet = [(findFeatures(rev), category) for (rev, category) in documents]

	print("NB_Train: Creating NaiveBayesClassifier. Training.")
	classifier = nltk.NaiveBayesClassifier.train(featureSet)

	print("NB_Train: Accuracy percent: ", (nltk.classify.accuracy(classifier, featureSet)) * 100)

	print("NB_Train: Saving classifier.")
	fileObj = open('NB_Classifier.pickle', 'wb')
	pickle.dump(classifier, fileObj)
	fileObj.close()

	print("NB_Train: Done.")


def CleanStringAndExtractHashtags(string):
	print("CleanStringAndExtractHashtags: Starting. String received:", string)

	def WordTokenizeString(string):
		return word_tokenize(string)

	def TweetTokenizeString(string):
		tweetTokenizer = TweetTokenizer(strip_handles = True, reduce_len = True)
		return " ".join(tweetTokenizer.tokenize(string))

	def RemoveApostropheWords(wordList):
		apostropheWords = {
							"n't":"not",
						  	"'s": ["is", "us"],
						  	"'ll": "will",
						  	"'ve": "have",
						  	"'re": "are",
						  	"'d": "would",
						  	"'m": "am",
						  	"c'mon": ["come", "on"],
						  	"y'all": ["you", "all"]
						  }

		removedApostropheWords = []

		previousWord = None
		for word in wordList:
			if (word in apostropheWords):
				if (word == "'s"):
					index = int(previousWord != "it")
					removedApostropheWords.append(apostropheWords[word][index])
				elif (len(apostropheWords[word]) > 1):
					removedApostropheWords.extend(apostropheWords[word])
				else:
					removedApostropheWords.append(apostropheWords[word])
			else:
				removedApostropheWords.append(word)
			previousWord = word

		return removedApostropheWords

	def LemmatizeWords(wordList):
		POSTaggedWordList = pos_tag(wordList)

		lemmatizedWords = []

		wordNetLemmatizer = WordNetLemmatizer()

		for word, partOfSpeech in POSTaggedWordList:

			try:
				lemmatizedWord = wordnet.morphy(word, pos=partOfSpeech[0].lower())
			except:
				lemmatizedWord = wordnet.morphy(word)
						
			if (lemmatizedWord == None):
				try:
					lemmatizedWord = wordNetLemmatizer.lemmatize(word, pos=partOfSpeech[0].lower())
				except:
					lemmatizedWord = wordNetLemmatizer.lemmatize(word)

			lemmatizedWords.append(lemmatizedWord)

		return lemmatizedWords

	# http://norvig.com/spell-correct.html
	def CorrectWords(wordList):
		
		global WORDS

		def words(text):
			return re.findall(r'\w+', text.lower())

		if(WORDS == None):
			WORDS = Counter(words(open('Glove_BagOfWords.txt').read()))

		def P(word, N=sum(WORDS.values())): 
			"Probability of `word`."
			return WORDS[word] / N

		def correction(word):
			"Most probable spelling correction for word."
			return max(candidates(word), key=P)

		def candidates(word): 
			"Generate possible spelling corrections for word."
			return (known([word]) or known(edits1(word)) or known(edits2(word)) or [word])

		def known(words): 
			"The subset of `words` that appear in the dictionary of WORDS."
			return set(w for w in words if w in WORDS)

		def edits1(word):
			"All edits that are one edit away from `word`."
			letters    = 'abcdefghijklmnopqrstuvwxyz'
			splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
			deletes    = [L + R[1:]               for L, R in splits if R]
			transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
			replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
			inserts    = [L + c + R               for L, R in splits for c in letters]
			return set(deletes + transposes + replaces + inserts)

		def edits2(word): 
			"All edits that are two edits away from `word`."
			return (e2 for e1 in edits1(word) for e2 in edits1(e1))

		correctedWords = []

		for word in wordList:
			correctedWord = correction(word)
			if (len(correctedWord) > 0):
				correctedWords.append(correctedWord)
			else:
				correctedWords.append(word)

		return correctedWords

	def RemoveStopwords(wordList):
		stopWords = set(stopwords.words("english"))
		return [word for word in wordList if word not in stopWords]

	def RemoveURLs(string):
		result = re.sub(r"http\S+", "", string)
		return result

	def ReplaceAbbreviatedWords(wordList):
		global abbreviationsDict

		if (len(abbreviationsDict) == 0):
			with open('abbreviations.csv', mode='r') as infile:
				reader = csv.reader(infile)
				for rows in reader:
					rows[0] = rows[0].split("~")
					abbreviationsDict[rows[0][0].lower()] = rows[0][1].lower()

		newWordList = []

		for word in wordList:
			if word in abbreviationsDict:
				newWordList.extend(word_tokenize(abbreviationsDict[word]))
			else:
				newWordList.append(word)

		return newWordList

	def RemovePunctuations(wordList):
		string = " ".join(wordList)
		string = "".join(character for character in string.strip() if character not in punctuation)

		return word_tokenize(string)

	lowercaseString = string.lower()
	print("\tConverting string to lower case:", lowercaseString)

	removedURLString = RemoveURLs(lowercaseString)
	print("\tAfter removing URLs:", removedURLString)

	tweetTokenizedString = TweetTokenizeString(removedURLString)
	print("\tAfter tweet tokenizing:", tweetTokenizedString)

	hashtags = [word for word in tweetTokenizedString.split(' ') if '#' in word]
	print("\tExtracting hashtags:", hashtags)

	wordTokenizedString = WordTokenizeString(tweetTokenizedString)
	print("\tAfter word tokenizing:", wordTokenizedString)
	
	removedApostropheWords = RemoveApostropheWords(wordTokenizedString)
	print("\tAfter removing apostrophe words:", removedApostropheWords)

	replacedAbbreviatedWords = ReplaceAbbreviatedWords(removedApostropheWords)
	print("\tAfter replacing abbreviations:", replacedAbbreviatedWords)

	removedPunctuations = RemovePunctuations(replacedAbbreviatedWords)
	print("\tAfter removing punctuations:", removedPunctuations)

	lemmatizedWords = LemmatizeWords(removedPunctuations)
	print("\tAfter lemmatizing words:", lemmatizedWords)

	correctedWords = CorrectWords(lemmatizedWords)
	print("\tAfter correcting words to their correct spelling:", correctedWords)

	cleanedString = " ".join(correctedWords)
	print("\tCleaned string:", cleanedString)

	return cleanedString, hashtags

def BuildBuffers():
	print("BuildBuffers: Tweepy is initializing")

	auth = OAuthHandler(ckey, csecret)
	auth.set_access_token(atoken, asecret)
	twitterStream = Stream(auth, listener())
	twitterStream.filter(languages=["en"] ,track=["#"])


print("Main: Starting to listen.")
BuildBuffers()