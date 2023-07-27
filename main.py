import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

lemmatizer = WordNetLemmatizer()
intents_file = json.loads(open("intents.json").read())

words = pickle.load(open("words.pkl",'rb'))
classes = pickle.load(open("classes.pkl",'rb'))

model = load_model("chatbot_model.h5")

def process_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(words) for words in sentence_words]
    return sentence_words

# Creating a bag of words which is basically a list of classes where the word is present
def bag_words(sentence):
    sentence_words = process_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i,word in enumerate(words):
            if word == w:
                bag[i] = 1

    return np.array(bag)

def predict(sentence):
    bag = bag_words(sentence)
    res = model.predict(np.array([bag]),verbose = 0)[0]
    thresh = 0.25 # Error threshold
    result = [[i,r] for i,r in enumerate(res) if r > thresh]
    result.sort(key = lambda x : x[1], reverse= True)
    ret = []
    for r in result:
        ret.append({'intent' : classes[r[0]], 'probability' : str(r[1])})
    return ret

def generate_response(inp):
    tag = predict(inp)[0]['intent']
    intents_list = intents_file['intents']
    for i in intents_list:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

while True:
    inp = input("Ask me anything: ")
    if(inp == "exit"):
        break
    print("Pychat: ",generate_response(inp))
    print("\n")

