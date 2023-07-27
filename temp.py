# Importing packages to load and process data
import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer

# Run your program with the following lines if running for the first time
#nltk.download('punkt')
#nltk.download('wordnet')

# Importing Tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD

# Loading and processing the data
lemmatizer = WordNetLemmatizer()
intents_file = json.loads(open("intents.json").read())

# Use this stament to verify the file is being loaded
#print(intents_file)

words = []
classes = []
documents = []
ignore_letters = ['?','!','.','-','(',')',','] # Characters to be ignored while inputting data

# Loading the data into lists for training
for intent in intents_file["intents"]:
    for pattern in intent["patterns"]:
        words_list = nltk.word_tokenize(pattern) # Tokenize is used to seperate each word from a string
        words.extend(words_list) 
        documents.append((words_list,intent["tag"])) # Making a list with all patterns and their tags
        if intent["tag"] not in classes:
            classes.append(intent["tag"])

# Lemmatizing is reducing the word to the stem i.e working/works/worked -> work
words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
words = sorted(set(words))
classes = sorted(set(classes)) # Making sure no duplicates in classes and sorting it

# words is a sorted set of all the pattern pointers in the program
# classes are the tags of each intent
# documents is what maps each pattern to tag

# Making binary file for storing words and classes using pickle
pickle.dump(words,open("words.pkl",'wb'))
pickle.dump(classes,open("classes.pkl",'wb'))

training = []
output_empty = [0] * len(classes)

for document in documents:
    bag = [] # Is used to store the values of words which are a pattern of a class
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)

    output_row = list(output_empty) # Is the particular class that the words are a part of
    output_row[classes.index(document[1])] = 1
    # Traning is now a list of all inputs(bags) and their labels(output_row)
    training.append(bag  + output_row)

random.shuffle(training)    
training = np.array(training)

# Features and their labels
train_x = training[:,:len(words)]
train_y = training[:,len(words):]
#print(train_x[0])

# Neural Network
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss = "categorical_crossentropy",optimizer = sgd, metrics = ['accuracy'])

# Next line is used to train the model higher the epoch count higher the accuracy
temp = model.fit(train_x, train_y, epochs = 200, batch_size = 5, verbose = 1)

model.save("chatbot_model.h5",temp)