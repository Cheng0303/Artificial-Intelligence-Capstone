import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split

class LSTMModel:

    def __init__(self, df):
        self.df = df
        self.X_test = None
        self.y_test = None
        self.model = None
    
    def train(self):
        print(tf.__version__)
        print(df.shape)
        # Convert Sentiment Labels to Numbers
        self.df["Sentiment_Label"] = self.df["Sentiment_Label"].map({"Negative": 0, "Neutral": 1, "Positive": 2})

        # Tokenize Text
        max_words = 5000  # Max number of words in the vocabulary
        max_length = 50   # Max sequence length

        tokenizer = Tokenizer(num_words=max_words)
        tokenizer.fit_on_texts(self.df["Body"])
        X = tokenizer.texts_to_sequences(self.df["Body"])
        X = pad_sequences(X, maxlen=max_length)
        y = self.df["Sentiment_Label"].values

       
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
        self.X_test = X_test
        self.y_test = y_test

        model = Sequential([
            Embedding(input_dim=max_words, output_dim=128, input_length=max_length),  # Word embeddings
            LSTM(64, return_sequences=True),  # First LSTM layer
            LSTM(32),  # Second LSTM layer
            Dropout(0.5),  # Prevent overfitting
            Dense(3, activation="softmax")  # 3 sentiment classes (Negative, Neutral, Positive)
        ])

        # Compile Model
        model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

        # Train Model with Validation
        history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=5, batch_size=32)
        self.model = model

        # Save Model
        model.save("lstm_sentiment_model.h5")

    def predict(self):
        self.model.evaluate(self.X_test, self.y_test)



