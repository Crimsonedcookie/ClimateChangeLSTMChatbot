import json
import keras
import numpy as np
from config_training_data import sample

word_model = keras.models.load_model("/Users/nathanramani/PycharmProjects/ClimateChangeChatbot/models/word_lstm/v1/word_lstm_v1.keras")
with open("/Users/nathanramani/PycharmProjects/ClimateChangeChatbot/models/word_lstm/v1/word_vocabulary.json", "r", encoding="utf-8") as f:
    vocabulary=json.load(f)
word_indices = {word: i for i, word in enumerate(vocabulary)}
indices_word = {i: word for i, word in enumerate(vocabulary)}


character_model = keras.models.load_model("/Users/nathanramani/PycharmProjects/ClimateChangeChatbot/models/character_lstm/v1/character_lstm_v1.keras")
with open("/Users/nathanramani/PycharmProjects/ClimateChangeChatbot/models/character_lstm/v1/climate_vocab.json", "r", encoding="utf-8") as f:
    chars=json.load(f)
char_indices={c:i for i, c in enumerate(chars)}
indices_char={i:c for i, c in enumerate(chars)}

maxlen = 40

def generate_character_response(starttext, lengthcharacters=200, diversity=0.5):
    starttext=starttext.lower().strip()
    #pad/truncate to maxlen
    if len(starttext) < maxlen:
        starttext= " " * (maxlen - len(starttext)) + starttext
    else:
        starttext=starttext[-maxlen:]
    generated= ""
    sentence=starttext
    for i in range(lengthcharacters):
        x_pred=np.zeros((1, maxlen, len(chars)))
        for t, char in enumerate(sentence):
            if char in char_indices:
                x_pred[0, t, char_indices[char]]=1.0
        preds = character_model.predict(x_pred, verbose=0)[0]
        next_index=sample(preds, diversity)
        next_char=indices_char[next_index]
        sentence=sentence[1:]+next_char
        generated+=next_char
    return generated

def generate_word_response(starttext, lengthwords=200, diversity=0.5):
    words=starttext.lower().strip().split()
    #Keep last 40 words
    words=words[-maxlen:]
    generated=[]
    for i in range(lengthwords):
        #Create input sequence
        x_pred=np.zeros((1, maxlen), dtype=np.int32)
        #Maxlen positions
        input_words=words[-maxlen:]
        #Pad left
        start_position=maxlen - len(input_words)
        for t, word in enumerate(input_words):
            if word in word_indices:
                x_pred[0, start_position + t]=word_indices[word]
        #Predict next word
        preds=word_model.predict(x_pred, verbose=0)[0]
        next_index=sample(preds, diversity)
        next_word=indices_word[next_index]
        words.append(next_word)
        generated.append(next_word)
    return " ".join(generated)


def generate_response(starttext, model_type, length, diversity=0.5):
    if model_type=="character":
        return generate_character_response(starttext, lengthcharacters=length, diversity=diversity)
    elif model_type=="word":
        return generate_word_response(starttext, lengthwords=length, diversity=diversity)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

"""TEMPORARY VERSION for word_lstm_v1. Need to make this pipeline compatible with both models, but that'll come afterwards."""

# import keras
# import numpy as np
# import json
#
#
# MODEL_PATH = "/Users/nathanramani/PycharmProjects/ClimateChangeChatbot/models/word_lstm/v1/word_lstm_v1.keras"
# VOCAB_PATH = "/Users/nathanramani/PycharmProjects/ClimateChangeChatbot/models/word_lstm/v1/word_vocabulary.json"
#
# maxlen = 40
#
# model = keras.models.load_model(MODEL_PATH)
#
# with open(VOCAB_PATH, "r", encoding="utf-8") as f:
#     vocabulary = json.load(f)
#
# word_indices = {word: i for i, word in enumerate(vocabulary)}
# indices_word = {i: word for i, word in enumerate(vocabulary)}
#
#
# def sample(preds, temperature=1.0):
#     preds = np.asarray(preds).astype("float64")
#     preds = np.log(np.maximum(preds, 1e-10)) / temperature
#
#     exp_preds = np.exp(preds)
#     preds = exp_preds / np.sum(exp_preds)
#
#     probas = np.random.multinomial(1, preds, 1)
#
#     return np.argmax(probas)
#
#
# def generate_response(starttext, lengthwords=50, diversity=0.5):
#
#     words = starttext.lower().strip().split()
#
#     # Keep only the last 40 words
#     words = words[-maxlen:]
#
#     generated = []
#
#     for i in range(lengthwords):
#
#         # Create integer input sequence
#         x_pred = np.zeros((1, maxlen), dtype=np.int32)
#
#         # We need exactly maxlen positions
#         input_words = words[-maxlen:]
#
#         # Pad on the left if necessary
#         start_position = maxlen - len(input_words)
#
#         for t, word in enumerate(input_words):
#             if word in word_indices:
#                 x_pred[0, start_position + t] = word_indices[word]
#
#         # Predict next word
#         preds = model.predict(x_pred, verbose=0)[0]
#
#         next_index = sample(preds, diversity)
#         next_word = indices_word[next_index]
#
#         words.append(next_word)
#         generated.append(next_word)
#
#     return " ".join(generated)