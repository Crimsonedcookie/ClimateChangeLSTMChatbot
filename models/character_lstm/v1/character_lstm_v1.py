import keras
from keras import layers
import numpy as np
import io
import json
from keras.callbacks import EarlyStopping, ReduceLROnPlateau


#get dataset ready

path=keras.utils.get_file('climatechangeinfofile.txt','file:///Users/nathanramani/Downloads/climatechangeinfofile .txt')

with io.open(path, encoding="utf-8") as f:
    text=f.read().lower()
text=text.replace("\n", " ")

print("Corpus length:", len(text))
chars=sorted(list(set(text)))
print("Total chars:", len(chars))
char_indices=dict((c, i) for i, c in enumerate(chars))
indices_char=dict((i, c) for i, c in enumerate(chars))

#create sentences for training

maxlen=40
step=3
sentences=[]
next_chars=[]

for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i:i + maxlen])
    next_chars.append(text[i + maxlen])
print("Number of sequences:", len(sentences))

x=np.zeros((len(sentences), maxlen, len(chars)), dtype=bool)
y=np.zeros((len(sentences), len(chars)), dtype=bool)

for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]]=1
    y[i, char_indices[next_chars[i]]]=1

#building and training model

model=keras.Sequential([keras.Input(shape=(maxlen, len(chars))), layers.LSTM(128), layers.Dense(len(chars), activation="softmax")])
model.compile(loss="categorical_crossentropy", optimizer=keras.optimizers.Adam(learning_rate=0.001), metrics=["accuracy"])
callbacks=[EarlyStopping(monitor="val_loss", patience=50, restore_best_weights=True), ReduceLROnPlateau(monitor="val_accuracy", factor=0.5, patience=3)]

epochs = 200
batch_size = 128
print(model.summary())

model.fit(x, y, epochs=epochs, batch_size=batch_size, validation_split=0.2, callbacks=callbacks)

#saving model :D

model.save("climate_lstm.keras")
with open("climate_vocab.json", "w", encoding="utf-8") as f:
    json.dump(chars, f)