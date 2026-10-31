import keras
from keras import layers
import numpy as np
import io
from collections import Counter
import json
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from pathlib import Path



"""This model has the same general architecture as character_lstm_v1, with a few exceptions:
        1. Predicting words, not characters (obviously)
        2. Encoding using just integers, not one-hot encoding
        3. Added embedding layer to the model
        4. Compiling model with sparse categorical crossentropy, not regular because of the new encoding
"""

#get dataset ready

model_dir=Path(__file__).resolve().parent
path=keras.utils.get_file('climatechangeinfofile.txt','file:///Users/nathanramani/Downloads/climatechangeinfofile.txt')

with io.open(path, encoding="utf-8") as f:
    text=f.read().lower()
text=text.replace("\n", " ")

words=text.split()
word_counts=Counter(words)
vocabulary=sorted(word_counts.keys())
vocab_size=len(vocabulary)

word_indices=dict((c, i) for i, c in enumerate(vocabulary))
indices_word=dict((i, c) for i, c in enumerate(vocabulary))

#create sentences for training

maxlen=40
step=3
sentences=[]
next_words=[]

for i in range(0, len(words) - maxlen, step):
    sentences.append(words[i:i + maxlen])
    next_words.append(words[i + maxlen])
print("Number of sequences:", len(sentences))


#sentence encoding

x=np.zeros((len(sentences), maxlen), dtype=np.int32)
y=np.zeros((len(sentences),), dtype=np.int32)

for i, sentence in enumerate(sentences):
    for t, word in enumerate(sentence):
        x[i,t]=word_indices[word]
    y[i]=word_indices[next_words[i]]


#building model
embedding_dim=128
model=keras.Sequential([keras.Input(shape=(maxlen,)), layers.Embedding(vocab_size,embedding_dim), layers.LSTM(128), layers.Dense(len(vocabulary), activation="softmax")])
model.compile(loss="sparse_categorical_crossentropy", optimizer=keras.optimizers.Adam(learning_rate=0.001), metrics=["accuracy"])


#training model
callbacks=[EarlyStopping(monitor="val_loss", patience=50, restore_best_weights=True), ReduceLROnPlateau(monitor="val_accuracy", factor=0.5, patience=3)]

epochs = 200
batch_size = 128
print(model.summary())

model.fit(x, y, epochs=epochs, batch_size=batch_size, validation_split=0.2, callbacks=callbacks)


#saving model :D

model.save(model_dir/"word_lstm.keras")
with open(model_dir/"word_vocabulary.json", "w", encoding="utf-8") as f:
    json.dump(vocabulary, f)