import nltk

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

import json
import pickle
import random
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD

word=[]
clas=[]
documents=[]
ignore_words=['.','@','?','$']
data_file = open('intents.json').read()
intents = json.loads(data_file)

for intent in intents['intents']:
	for pattern in intent['patterns']:
		w=nltk.word_tokenize(pattern)
		word.extend(w)
		documents.append((w, intent['tag']))
		if intent['tag'] not in clas:
			clas.append(intent['tag'])

word = [lemmatizer.lemmatize(w.lower()) for w in word if w not in ignore_words]
word = list(set(word))
clas = list(set(clas))

pickle.dump(word, open('word.pkl', 'wb'))
pickle.dump(clas, open('clas.pkl', 'wb'))

training = []
output_empty = [0]*len(clas)

for doc in documents:
	bag = []
	pattern_words = doc[0]
	pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]

	for w in word:
		bag.append(1) if w in pattern_words else bag.append(0)

	output_row = list(output_empty)
	output_row[clas.index(doc[1])] = 1
	training.append([bag, output_row])

random.shuffle(training)
training = np.array(training)

train_x = list(training[:, 0])
train_y = list(training[:, 1])
print('X: {}'.format(train_x))
print('Y: {}'.format(train_y))

model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',optimizer=sgd, metrics=['accuracy'])

mfit = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('model.h5', mfit)

print('Model created successfully!!!!!!!!!!!!!')
