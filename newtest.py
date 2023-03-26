# Name: Shiva Kumar
# Net id: sak220007
# Course: CS 6320


import tensorflow as tf
import os
import numpy as np
import collections
from keras.utils import pad_sequences
from keras.models import Sequential
from keras.layers import InputLayer, Activation
from keras.layers import Dense, LSTM, InputLayer, Bidirectional, TimeDistributed, Embedding
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras import losses
from keras import optimizers
from keras import metrics
from keras.losses import binary_crossentropy
from keras.losses import mean_squared_error

from sklearn.preprocessing import OneHotEncoder
# Description of program: The program has two part-of-speech-taggers(POST):
# 1. Using Hidden Markov Models (HMM)
# 2. Using Recurrent Neural Network (RNN)

# HMMTagger class is a POST using HMM


class HMMTagger():
    def __init__(self):
        self.initialProbabilities = {}
        self.transitionCounts = collections.defaultdict(dict)
        self.transitionSmoothingProbabilities = collections.defaultdict(dict)
        self.emissionCounts = collections.defaultdict(dict)
        self.emissionSmoothingProbabilities = collections.defaultdict(
            dict)

        # Keeps frequency of tags
        self.unigramTagCount = {}

        # Set of all tags in corpus
        self.Tags = set()
        # Set of all words in corpus
        self.Tokens = set()

    # load_corpus:
    # Input: Path to the Corpus Folder
    # Output: Returns the corpus has a list of POS-tagged sentences

    def load_corpus(self, path):
        # Contains array of all file arrays. This array will be retured
        Final_Corpus = []
        # Loops through every file in Corpus Folder
        for file in os.listdir(path):
            # Creating the path for currentFile
            FilePath = path + file

            # Contains array of final sentences
            FileArray = []

            # Reads the data in the FilePath
            with open(FilePath, 'r') as FilePath:
                data = FilePath.readlines()

                for sentence in data:
                    # Converts sentence based on white-space into an array
                    sentence = sentence.split()

                    # Is the Final Sentence array
                    FinalSentence = []

                    for value in sentence:
                        splitIndex = value.index("/")
                        FinalValue = (value[:splitIndex], value[splitIndex+1:])
                        FinalSentence.append(FinalValue)
                    # Question to ask: How should blank sentences be handled?
                    if len(FinalSentence) > 0:
                        FileArray.append(FinalSentence)
            Final_Corpus.append(FileArray)
        return Final_Corpus
    # initialize_probabilities compute initial probabilities, transition probabilies and
    # emission probabilities
    # Input: For every file of the train corpus, there is an array of POS sentence arrays.
    # So, that for every sentence an array with (word, POS) items.

    def initialize_probabilities(self, POSSentenceListCorpus):

        # Transition Probability

        # Keeps track of the total sentences
        totalSentences = 0

        # Goes through every file POS Setentence list
        for file in POSSentenceListCorpus:

            # For every POS sentence List in current file
            for sentence in file:
                # Updates totalSentences
                totalSentences += 1

                # POSPairIndex is (word,POS) pair
                for POSPairIndex in range(len(sentence)):
                    # currentTag represents the current tag
                    currentTag = sentence[POSPairIndex][1]
                    # currentWord represents the current word
                    currentWord = sentence[POSPairIndex][0]

                    # lower case currentWord
                    currentWord = currentWord.lower()
                    # If currentTag is a new tag
                    if currentTag not in self.Tags:
                        self.Tags.add(currentTag)
                    # If currentWord is a new word
                    if currentWord not in self.Tokens:
                        self.Tokens.add(currentWord)
                    # Keeps track of the frequency of each tag in UnigramTagCount
                    if currentTag not in self.unigramTagCount:
                        self.unigramTagCount[currentTag] = 1
                    else:
                        self.unigramTagCount[currentTag] += 1
                    # Start of Sentence represents to update initalProbabilities
                    if POSPairIndex == 0:
                        # If the currentTag is not in initialProbabilites
                        if currentTag not in self.initialProbabilities:
                            self.initialProbabilities[currentTag] = 1
                        # Update count if currentTag is in initalProbabilites
                        else:
                            self.initialProbabilities[currentTag] += 1

                    # Not start of sentence
                    else:
                        # previousTag represents the previous tag
                        previousTag = sentence[POSPairIndex-1][1]
                        # Current Tag has not been after previous tag
                        if currentTag not in self.transitionCounts[previousTag]:
                            self.transitionCounts[previousTag][currentTag] = 1
                        else:
                            self.transitionCounts[previousTag][currentTag] += 1
                        # Current Word has not been with currnetTag
                        if currentWord not in self.emissionCounts[currentTag]:
                            self.emissionCounts[currentTag][currentWord] = 1
                            # Initialize Unseen data.
                            # Unseen will be used when there is a word in the test set
                            # but not in the training corpus
                            self.emissionCounts[currentTag]["unseen"] = 0
                        else:
                            # Update count if currentWord has been after previous tag
                            self.emissionCounts[currentTag][currentWord] += 1

        # initalize probability is (probability + 1)/ (totalSentences + number of tags)
        for tag in self.initialProbabilities.keys():

            self.initialProbabilities[tag] = (
                self.initialProbabilities[tag] + 1) / (len(self.Tags)+totalSentences)
        # For every previous Tag dictionary in the self.transitionCounts dictionary
        for previousTag in self.transitionCounts.keys():
            # For every Tag following previousTag
            for FollowTag in self.transitionCounts[previousTag].keys():
                self.transitionSmoothingProbabilities[previousTag][FollowTag] = (
                    self.transitionCounts[previousTag][FollowTag] + 1) / (self.unigramTagCount[previousTag] + len(self.Tags))

        # For every currentTag dictionary in the self.emissionCounts dictionary
        for currentTag in self.emissionCounts.keys():
            # For every word following currentTag
            for FollowWord in self.emissionCounts[currentTag].keys():
                # The extra 1 in the denominator represents unseen data in the test set
                self.emissionSmoothingProbabilities[currentTag][FollowWord] = (
                    self.emissionCounts[currentTag][FollowWord] + 1) / (self.unigramTagCount[currentTag] + len(self.Tokens) + 1)

    # Implement viterbi algo which implements Dynamic Programming

    def viterbi_decode(self, testSentence, virterbiAnswer, index, previousTag):

        if (index == len(testSentence)):
            return virterbiAnswer

        # word we are comparing
        word = testSentence[index]

        # Initailize Max Probability
        # Will use to compare viterbiAnswers
        maxViterebiProbability = 0

        ViterbiTag = "None"
        # lower case word
        word = word.lower()

        # Loops through every tag
        for currentTag in self.Tags:

            # Represents current viterbi probability
            currentViterbiProbability = 0
            # Probability of word given tag
            if word in self.emissionSmoothingProbabilities[currentTag].keys():
                emissionProb = self.emissionSmoothingProbabilities[currentTag][word]
            else:
                emissionProb = self.emissionSmoothingProbabilities[currentTag]["unseen"]
            # For every tag in initialProbabilites compute viterbi probability
            if (index == 0):
                # Probability of tag starting the sentence
                initalProb = self.initialProbabilities[currentTag]
                currentViterbiProbability = emissionProb * initalProb

            # If the word is not the start of the sentence then use transition probabilites
            else:

                # Probability of current tag given previous tag
                transitionProbability = self.transitionSmoothingProbabilities[
                    previousTag][currentTag]

                # Update viterbiMatrix
                currentViterbiProbability = emissionProb * transitionProbability
            if currentViterbiProbability > maxViterebiProbability:
                maxViterebiProbability = currentViterbiProbability
                ViterbiTag = currentTag

        # Add Tag to answer array
        virterbiAnswer.append(ViterbiTag)
        # Recursive call
        return self.viterbi_decode(testSentence,
                                   virterbiAnswer, index+1, virterbiAnswer[index])


class RNN():
    def __init__(self):
        # tokenDictionary is the dictionary that contains integer key for train_X : word
        self.tokenDictionary = {}
        # tagDictionary is the dictionary that contains integer key for train_Y: tag
        self.tagDictionary = {}

        # Defines the relevant lists.
        self.train_X = list()
        self.train_y = list()

        # Max Length of the biggest list in train_X
        self.MAX_LENGTH = 0

        self.hotEncodeTags = {}
        # train_Y with all the Hot One Encode Tags
        self.CategoricalTrainy = []
        self.numTagDictionary = {}
        # Has numTag has keys and the Tag index has values
        self.numTagtoIndex = {}

        self.originalSize = 0
        self.ReverseCategorical = {}
    # load_corpus:
    # Input: Path to the Corpus Folder
    # Output: Returns the corpus has a list of POS-tagged sentences

    def load_corpus(self, path):
        # Contains array of all file arrays. This array will be retured
        Final_Corpus = []
        # Loops through every file in Corpus Folder
        for file in os.listdir(path):
            # Creating the path for currentFile
            FilePath = path + file

            # Contains array of final sentences
            FileArray = []

            # Reads the data in the FilePath
            with open(FilePath, 'r') as FilePath:
                data = FilePath.readlines()

                for sentence in data:
                    # Converts sentence based on white-space into an array
                    sentence = sentence.split()

                    # Is the Final Sentence array
                    FinalSentence = []

                    for value in sentence:
                        splitIndex = value.index("/")
                        FinalValue = (value[:splitIndex], value[splitIndex+1:])
                        FinalSentence.append(FinalValue)
                    # Question to ask: How should blank sentences be handled?
                    if len(FinalSentence) > 0:
                        FileArray.append(FinalSentence)

            Final_Corpus.append(FileArray)
        return Final_Corpus

    # Creates the dataset with train_X (words) and train_y (tag).
    def create_dataset(self, POSSentenceListCorpus):

        # At integer 0 for tokenDictionary will represent  PAD
        self.tokenDictionary["PAD"] = 0
        self.tagDictionary["PAD"] = 0
        # For every file in the trainingCorpus

        wordIndex = 1
        TagIndex = 1

        for file in POSSentenceListCorpus:
            # For every sentence in the file
            for sentence in file:
                currenttrain_X = list()
                currenttrain_y = list()
                for POSPair in sentence:

                    # Represents the current word in the POS pair
                    currentWord = POSPair[0]

                    # Lower case each word
                    currentWord = currentWord.lower()

                    # Represents the current tag in the POS pair
                    currentTag = POSPair[1]

                    if currentWord not in self.tokenDictionary.keys():
                        self.tokenDictionary[currentWord] = wordIndex

                        wordIndex += 1
                    if currentTag not in self.tagDictionary.keys():
                        self.tagDictionary[currentTag] = TagIndex

                        TagIndex += 1
                    currenttrain_X.append(currentWord)
                    currenttrain_y.append(currentTag)

                indexArrayX = []
                for word in currenttrain_X:
                    indexArrayX.append(self.tokenDictionary[word])
                # Add currenttrain_X to self.train_X
                self.train_X.append(np.array(indexArrayX))

                indexArrayY = []
                for word in currenttrain_y:
                    indexArrayY.append(self.tagDictionary[word])
                # Add currenttrain_y to self.train_y
                self.train_y.append(np.array(indexArrayY))

        # Convert train_X,train_Y List to numpy array
        self.train_X = np.array(self.train_X, dtype="object")
        self.train_y = np.array(self.train_y, dtype="object")

        return
    # Pad the sequences with 0s to the max length.

    def pad_sequences(self):

        # Use MAX_LENGTH to record length of longest sequence

        self.MAX_LENGTH = len(max(self.train_X, key=len))
        # Padding 0's using keras for both train_X and train_y
        self.train_X = tf.keras.preprocessing.sequence.pad_sequences(
            self.train_X, maxlen=self.MAX_LENGTH, padding='post')

        self.train_y = tf.keras.preprocessing.sequence.pad_sequences(
            self.train_y, maxlen=self.MAX_LENGTH, padding='post')

        return

    def define_model(self, numTags=11):
        # Sequential model
        Model = tf.keras.Sequential()

        # Calculates how many words are in training corpus
        VocabSize = 0
        for key in self.tokenDictionary.keys():
            VocabSize += 1

        Model.add(InputLayer(input_shape=(self.MAX_LENGTH, )))

        # Size of vectors are 128
        Model.add(Embedding(VocabSize, 128))
        # Hidden layers are 256
        Model.add(tf.keras.layers.LSTM(256, return_sequences=True))

        Model.add(TimeDistributed(Dense(numTags+1)))

        Model.summary()

        return Model

    def to_categorical(self, numTags=11):
        # Hot Encodes train y array
        self.train_y = to_categorical(self.train_y, numTags+1)

    def train(self, model):

        # Compiles the model with
        model.compile(loss='binary_crossentropy',
                      optimizer='adam', metrics=['accuracy'])
        # Fit Model for train_X and train_y
        # 40 Iterations
        model.fit(self.train_X, self.train_y, batch_size=128,
                  epochs=40, validation_split=0.2)
        return model

    def test(self, model, sentence):

        # Split the sentence
        sentence = sentence.split()
        # Array of Test Sentence with the unique integers
        newsentenceArray = []
        # For every word input the unique word index
        for word in sentence:
            sentenceArray = []
            # If word is not found
            if word not in self.tokenDictionary:
                sentenceArray.append(2)
            # If word is found
            else:
                sentenceArray.append(self.tokenDictionary[word])
            newsentenceArray.append(sentenceArray)

        # Padding values for the test sentence
        newsentenceArray = tf.keras.preprocessing.sequence.pad_sequences(
            newsentenceArray, maxlen=self.MAX_LENGTH, padding='post')
        # Predicts the new sentence
        outputEncode = model.predict(newsentenceArray)
        # NewOutput represents all the argmax values with the padded values
        NewOutput = []

        output = []
        # Reverse Key is used to find the Tag based on the argmax value
        ReverseKey = {}
        for current in self.tagDictionary.keys():
            value = self.tagDictionary[current]
            ReverseKey[value] = current

        # For every word in test sentence
        for category in outputEncode:
            output = []
            # For each word calculate argmax ( This will have lots of values becuase of the pad sequence method)
            for sequence in category:
                output.append(ReverseKey[np.argmax(sequence)])
            NewOutput.append(output)
        # Take first Index of FinalOutput has Result to get rid of PAD values
        FinalOutput = []
        for output in NewOutput:
            FinalOutput.append(output[0])
        return FinalOutput


# Input Test Sentences
testSentence1 = "the planet jupiter and its moons are in effect a mini solar system ."
testSentence2 = "computers process programs accurately ."


# Array of Input Sentences
# Note: If you want more test Sentences, define sentence above and add to testSentenceArray
testSentenceArray = [testSentence1, testSentence2]

# Corpus Path. If Path is named different folder then change the CorpusPath variable
CorpusPath = "./modified_brown/"

# Set up HMM Model
HMMModel = HMMTagger()
POSSentenceListCorpus = HMMModel.load_corpus(CorpusPath)
HMMModel.initialize_probabilities(POSSentenceListCorpus)

# Loop through every test sentence and call viterbi_decode for HMM Ouput
for testSentence in testSentenceArray:
    # Splits sentence by space and converts the testSentence into array
    testSentence = testSentence.split()
    POSArray = HMMModel.viterbi_decode(
        testSentence, [], 0, 'start')
    print("HMM POS TAG")
    print(POSArray)

# Set up RNN

RNNModel = RNN()
RNNTrainCorpus = RNNModel.load_corpus(CorpusPath)
RNNModel.create_dataset(RNNTrainCorpus)

RNNModel.pad_sequences()
Model = RNNModel.define_model()
RNNModel.to_categorical()

TrainedModel = RNNModel.train(Model)

for testSentence in testSentenceArray:
    print("Sentence ", testSentence)
    Output = RNNModel.test(TrainedModel, testSentence)
    print("RNN POS Tag")
    print("Output: ", Output)
