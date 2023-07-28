# Importing packages to load and process data
import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
import tensorflow as tf

# Run your program with the following lines if running for the first time
nltk.download('punkt',quiet=True)
nltk.download('wordnet',quiet=True)

word_lemmatizer = WordNetLemmatizer()

intents_file = json.loads(open('intents.json').read())

# Use this stament to verify the file is being loaded
#print(intents_file)

pattern_words, label, class_set = [],[],[]
ignore_lis = ['?', '!', '.', ',']

# Loading the data into lists for training
for intent in intents_file['intents']:
    for pattern in intent['patterns']:
        wordList = nltk.word_tokenize(pattern) # Tokenize is used to seperate each word from a string
        pattern_words.extend(wordList)
        class_set.append((wordList, intent['tag'])) # Making a list with all patterns and their tags
        if intent['tag'] not in label:
            label.append(intent['tag'])

# Lemmatizing is reducing the word to the stem i.e working/works/worked -> work
pattern_words = [word_lemmatizer.lemmatize(word) for word in pattern_words if word not in ignore_lis]
pattern_words = sorted(set(pattern_words))

labels = sorted(set(label))

# Making binary file for storing words and labels using pickle
pickle.dump(pattern_words, open('words.pkl', 'wb'))
pickle.dump(label, open('classes.pkl', 'wb'))

training_set = []
outputEmpty = [0] * len(labels)

for document in class_set:
    word_bag = [] # Is used to store the values of words which are a pattern of a class
    wordPatterns = document[0]
    wordPatterns = [word_lemmatizer.lemmatize(word.lower()) for word in wordPatterns]
    for word in pattern_words:
        word_bag.append(1) if word in wordPatterns else word_bag.append(0)

    outputRow = list(outputEmpty) # Is the particular class that the words are a part of
    outputRow[label.index(document[1])] = 1
    # Traning is now a list of all inputs(word_bag) and their labels(output_row)
    training_set.append(word_bag + outputRow)

random.shuffle(training_set)
training_set = np.array(training_set)

# Features and their labels
train_x = training_set[:, :len(pattern_words)]
train_y = training_set[:, len(pattern_words):]

#Neural network
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(128, input_shape=(len(train_x[0]),), activation = 'relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(64, activation = 'relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(len(train_y[0]), activation='softmax'))

# SGD stands for Stochastic Gradient Descent is used as an optimizer
sgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Next line is used to train the model higher the epoch count higher the accuracy
temp = model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)

# Saving the model values
model.save('chatbot_model.h5',temp)