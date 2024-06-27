import random
import nltk
import numpy as np
import json
import pickle
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('wordnet')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Load intents JSON file
with open('CovidQA.json') as json_file:
    intents = json.load(json_file)

# Load pre-trained model and supporting files
words = pickle.load(open('Words.pkl', 'rb'))
classes = pickle.load(open('Classes.pkl', 'rb'))
model = load_model('ChatbotModel.h5')

# Define global variable for correct question
correctQues = ""

# Define minimum edit distance function for spell correction
def minDis(s1, s2, n, m, dp):
    if n == 0:
        return m
    if m == 0:
        return n
    if dp[n][m] != -1:
        return dp[n][m]
    if s1[n - 1] == s2[m - 1]:
        if dp[n - 1][m - 1] == -1:
            dp[n][m] = minDis(s1, s2, n - 1, m - 1, dp)
            return dp[n][m]
        else:
            dp[n][m] = dp[n - 1][m - 1]
            return dp[n][m]
    else:
        if dp[n - 1][m] != -1:
            m1 = dp[n - 1][m]
        else:
            m1 = minDis(s1, s2, n - 1, m, dp)
        if dp[n][m - 1] != -1:
            m2 = dp[n][m - 1]
        else:
            m2 = minDis(s1, s2, n, m - 1, dp)
        if dp[n - 1][m - 1] != -1:
            m3 = dp[n - 1][m - 1]
        else:
            m3 = minDis(s1, s2, n - 1, m - 1, dp)
        dp[n][m] = 1 + min(m1, min(m2, m3))
        return dp[n][m]

# Define spell correction function
def correct_spelling(sentence):
    sentence_list = []
    for sent in sentence.split():
        word = ""
        dist = 5
        for w in words:
            n = len(sent)
            m = len(w)
            dp = [[-1 for _ in range(m + 1)] for _ in range(n + 1)]
            diff = minDis(sent, w, n, m, dp)
            if dist > diff:
                dist = diff
                word = w
        sentence_list.append(word)
    return ' '.join(sentence_list)

# Define function to clean up and tokenize the sentence
def clean_up_sentence(sentence):
    global correctQues
    sentence = correct_spelling(sentence)
    correctQues = sentence
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

# Define function to convert sentence into bag of words
def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

# Define function to predict class
def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = [{'intent': classes[r[0]], 'probability': str(r[1])} for r in results]
    return return_list

# Define function to get response
def get_response(intents_list, intents_json):
    global correctQues
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if tag in i['tags']:
            return random.choice(i['responses']), intents_json['intents'].index(i), tag
    return ["I do not know about this tags", -1, ""]

# Define main function
def main_(message: str):
    ints = predict_class(message)
    if len(ints) > 0:
        return get_response(ints, intents)
    else:
        return ["I do not know about it", -1, ""]

# Example usage
