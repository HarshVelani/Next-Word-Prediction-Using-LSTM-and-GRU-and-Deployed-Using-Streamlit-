import numpy as np
import pandas as pd
import streamlit as st
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Dense, LSTM, GRU, Embedding
import pickle

# Load the model
LSTM_model = load_model('LSTM_next_word_prediction.h5')
GRU_model = load_model('GRU_next_word_prediction.h5')

# Load the tokenizer
with open('LSTM_tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Function to predict next word
def predict_next_word(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len-1):] # ensures that sequence length matches the max sequence length
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted)

    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return None

# Streamlit app
st.title('Next Word Prediction')
st.write('This app predicts the next word in a sentence using LSTM and GRU models.')

model = st.selectbox('Select the model:', ['LSTM', 'GRU'])
input_text = st.text_input('Enter the input text:', 'To be or not to be')
n = st.slider('Number of words to predict:', 1, 10)

if model == 'LSTM':
    # n number of words to predict
    max_sequence_len = LSTM_model.input_shape[1]+1
    for _ in range(n):
        next_word = predict_next_word(GRU_model, tokenizer, input_text, max_sequence_len)
        input_text += ' ' + next_word
    st.write(f"Predicted text: {input_text}")
else:
    # n number of words to predict
    max_sequence_len = GRU_model.input_shape[1]+1
    for _ in range(n):
        next_word = predict_next_word(GRU_model, tokenizer, input_text, max_sequence_len)
        input_text += ' ' + next_word
    st.write(f"Predicted text: {input_text}")