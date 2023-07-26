# Importing packages to load and process data
import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer

# Run your program with the following line if running for the first time
#nltk.download('punkt')

# Importing Tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD

# Loading and processing the data
lemmatizer = WordNetLemmatizer
intents_file = json.loads(open("intents.json").read())

# Use this stament to verify the file is being loaded
#print(intents_file)

words = []
classes = []
documents = []
ignore_letters = ['?','!','.','\\','-','(',')'] # Characters to be ignored while inputting data

# Loading the data into lists for training
for intent in intents_file["intents"]:
    for pattern in intent["patterns"]:
        words_list = nltk.word_tokenize(pattern) # Tokenize is used to seperate each word from a string
        words.append(words_list) 
        documents.append(((words_list),intent["tag"])) # Making a list with all patterns and their tags
        if intent["tag"] not in classes:
            classes.append(intent["tag"])