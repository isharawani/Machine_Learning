from sre_parse import Tokenizer

import numpy as np
import pandas as  pd
import seaborn as sns
import matplotlib.pyplot as plt
df=pd.read_csv("quote_dataset.csv")
print(df.head())
print(df["quote"][0])

print(df.shape)
quotes=df["quote"].str.lower()
import string
translator=str.maketrans('','',string.punctuation)
quotes=quotes.apply(lambda x: x.translate(translator))
print(quotes)
print(quotes.head())
from tensorflow.keras.preprocessing.text import Tokenizer
vocab_size=10000
tokenizer=Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(quotes)
word_index=tokenizer.word_index
print("Number of unique tokens:",len(word_index))
print(list(word_index.items())[:10])
sequence=tokenizer.texts_to_sequences(quotes)
for i in range(3):
    print(quotes[0])
for i in range(3):
    print(sequence[i])

X=[]
y=[]
for seq in sequence:
    for i in range(1,len(seq)):
        input_seq=seq[:i]
        output_seq=seq[i]
        X.append(input_seq)
        y.append(output_seq)
   #print(len(X))
    #print(len(y))
from tensorflow.keras.preprocessing.sequence import pad_sequences

max_len=max(len(seq) for seq in X)
X=pad_sequences(X, maxlen=max_len, padding='pre')

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, SimpleRNN, Dense

embedding_dim=50
rnn_units=128
rnn_model=Sequential()
rnn_model.add(
    Embedding(input_dim=vocab_size,output_dim=embedding_dim,input_length=max_len)
)
rnn_model.add(SimpleRNN(units=rnn_units))
rnn_model.add(Dense(units=vocab_size, activation='softmax'))
rnn_model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)
print(rnn_model.summary())
 

lstm_model = Sequential()
lstm_model.add(
    Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len)
)
lstm_model.add(LSTM(units=rnn_units))
lstm_model.add(Dense(units=vocab_size, activation='softmax'))
     

lstm_model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
     

print(lstm_model.summary())

from tensorflow.keras.models import load_model

#lstm_model = load_model("lstm_model.h5")
#lstm_model.save("lstm_model.h5")
index_to_word = {}
for word, index in word_index.items():
  index_to_word[index] = word
     

from tensorflow.keras.preprocessing.sequence import pad_sequences
     

def predictor(model,tokenizer,text,max_len):
  text = text.lower()

  seq = tokenizer.texts_to_sequences([text])[0]
  seq = pad_sequences([seq], maxlen=max_len, padding='pre')

  pred = model.predict(seq,verbose = 0)
  pred_index = np.argmax(pred)
  return index_to_word[pred_index]

     

seed_text = "what are you"
next_word = predictor(lstm_model,tokenizer,seed_text,max_len)
print(next_word)
     
def generate_text(model,tokenizer,seed_text,max_len,n_words):
  for _ in range(n_words):
    next_word = predictor(model,tokenizer,seed_text,max_len)
    if next_word == "":
      break
    seed_text += " " + next_word
  return seed_text
     

seed = "are you a "
generate_text = generate_text(lstm_model,tokenizer,seed,max_len,10)
print(generate_text)
     


import pickle
with open("tokenizer.pkl", "wb") as f:
  pickle.dump(tokenizer, f)
     

with open("max_len.pkl", "wb") as f:
    print(pickle.dump(max_len, f))
     