import tkinter

from tkinter import *
from tensorflow.keras.models import load_model

import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

import json
import pickle
import random
import numpy as np

intents = json.loads(open('intents.json').read())
model = load_model('model.h5')
word = pickle.load(open('word.pkl', 'rb'))
clas = pickle.load(open('clas.pkl', 'rb'))

def bott(sentence):
	phrase = nltk.word_tokenize(sentence)
	phrase = [lemmatizer.lemmatize(word.lower()) for word in phrase]
	bag = [0]*len(word)
	for s in phrase:
		for i, w in enumerate(word):
			if w == s:
				bag[i]=1
	return (np.array(bag))

def predict(sentence):
	sentence_bag = bott(sentence)
	res = model.predict(np.array([sentence_bag]))[0]
	ERROR_THRESHOLD = 0.25
	results = [[i,r] for i,r in enumerate(res) if r > ERROR_THRESHOLD]
	results.sort(key=lambda x: x[1], reverse=True)
	return_list = []
	for r in results:
		return_list.append({'intent':clas[r[0]], 'probablity':str(r[1])})
	return return_list

def user_response(ints):
	tag = ints[0]['intent']
	list_of_intents = intents['intents']
	for i in list_of_intents:
		if(i['tag']==tag):
			result=random.choice(i['responses'])
			break

	return result

def chatbot_response(msg):
	ints = predict(msg)
	res = user_response(ints)
	return res

def send():
	msg = TextEntryBox.get("1.0", 'end-1c').strip()
	TextEntryBox.delete('1.0', 'end')

	if msg != '':
		ChatHistory.config(state=NORMAL)
		ChatHistory.insert('end', "You: " + msg + "\n\n")

		res = chatbot_response(msg)
		ChatHistory.insert('end', "Bot: " + res + "\n\n")
		ChatHistory.config(state=DISABLED)
		ChatHistory.yview('end')

def create():
	msg = TextEntryBox.get("1.0", 'end-1c').strip()
	TextEntryBox.delete('1.0', 'end')

	if msg != '':
		ChatHistory.config(state=NORMAL)
		#split string
		splits = msg.split()
		#for loop to iterate over words array
		ChatHistory.insert('end', "Your items : "+"\n")
		for split in splits:
			res = chatbot_response(split)
			ChatHistory.insert('end', split +" "+ res +"\n")
		ChatHistory.config(state=DISABLED)
		ChatHistory.yview('end')

base = Tk()
base.title("Supermarket_Bot")
base.geometry("400x500")
base.resizable(width=False, height=False)

#chat history textview
ChatHistory = Text(base, bd=0, bg='blue', font=('Arial', 12, 'bold'))
ChatHistory.config(state=DISABLED)

SendButton = Button(base, font=('Arial', 12, 'bold'),
	text="Send", bg="blue", activebackground="#3f3f3f", fg="#ffffff", command=send)
CreateListButton = Button(base, font=('Arial', 12, 'bold'),
	text="View", bg="blue", activebackground="#3e3e3e", fg="#ffffff", command=create)

TextEntryBox = Text(base, bd=0, bg='blue', font='Arial')

ChatHistory.place(x=6, y=6, height=386, width=386)
TextEntryBox.place(x=128, y=400, height=80, width=265)
SendButton.place(x=6, y=400, height=80, width=50)
CreateListButton.place(x=56, y=400, height=80, width=60)

base.mainloop()
