# Created a POS Tag using a Hidden Markov Model (HMM) and a recurrent neural network (RNN)
Shiva Kumar
CS 6320

Assumptions:
1. Lower Case and Upper-Case words are the same. 
Example: The and the are treated as the same words.

2.Using binary_crossentropy has my loss function for 
the model

3. Training Corpus will be in a folder called 
modified_brown.

Note: If this code is run on google collab then the 
path should be edited from “./modified_brown/” to 
“/modified_brown/”.

Have Keras, Tensorflow downloaded.

How to run my Program:
1. Put test sentences and edit test sentence array.

2. If run not on google collab:
Put all corpus files in folder called modified_brown.
In file, CorpusPath variable should be equal to 
“./modified_brown/”

3. Run Program

Note: Because the Model on Keras is 40 iterations 
running the entire program for RNN could take a 
couple of minutes. 

