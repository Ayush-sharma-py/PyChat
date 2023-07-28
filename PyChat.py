import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
import tensorflow


word_lemmatizer = WordNetLemmatizer()
intents_file = json.loads(open("intents.json").read())

pattern_words = pickle.load(open("words.pkl",'rb'))
classes = pickle.load(open("classes.pkl",'rb'))

# Loading the model which was generated using model.py
loaded_model = tensorflow.keras.models.load_model("chatbot_model.h5")

# Converts the sentence into the lemmatized form
def process_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [word_lemmatizer.lemmatize(words) for words in sentence_words]
    return sentence_words

# Creating a bag of words which is basically a list of classes where the word is present
def bag_words(sentence):
    sentence_words = process_sentence(sentence)
    bag = [0] * len(pattern_words)
    for w in sentence_words:
        for i,word in enumerate(pattern_words):
            if word == w:
                bag[i] = 1

    return np.array(bag)

# Getting the intent from the model
def predict(sentence):
    bag = bag_words(sentence)
    temp = loaded_model.predict(np.array([bag]),verbose = 0)[0]
    thresh = 0.25 # Error threshold
    result = [[i,r] for i,r in enumerate(temp) if r > thresh]
    result.sort(key = lambda x : x[1], reverse= True)
    ret = []
    for r in result:
        ret.append({'intent' : classes[r[0]], 'probability' : str(r[1])})
    return ret

# Using the predicted intent to generate a response
def generate_response(inp):
    tag = predict(inp)[0]['intent']
    intents_list = intents_file['intents']
    for i in intents_list:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

# Making a basic CLI
while True:
    inp = input("Ask me anything: ")
    if(inp == "exit"):
        break
    inp = inp.lower()
    print("Pychat: ",generate_response(inp))
    print("\n")

