import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE

class LSTMModel:

    def __init__(self, df):
        self.df = df
        self.X_test = None
        self.y_test = None
        self.model = None
        self.history = None
    
    def train(self):
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

        # Apply SMOTE for Class Imbalance
        # X, y = SMOTE(random_state=42).fit_resample(X, y)

       
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
        history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=3, batch_size=128)
        self.history = history
        self.model = model

        # Save Model
        model.save("lstm_sentiment_model.h5")

    def predict(self, X_test=None, y_test=None):
        if X_test is None:
            X = self.X_test
            y = self.y_test
        else:
            max_words = 5000  # Max number of words in the vocabulary
            max_length = 50   # Max sequence length

            tokenizer = Tokenizer(num_words=max_words)
            tokenizer.fit_on_texts(X_test["Body"])
            X = tokenizer.texts_to_sequences(X_test["Body"])
            X = pad_sequences(X, maxlen=max_length)
            y = y_test.values
            self.X_test = X
            self.y_test = y
        
        self.model.evaluate(X, y)
    
    def plot_loss(self):
        """Plot Training and Validation Loss"""
        if self.history:
            epochs_range = range(1, len(self.history.history["loss"]) + 1)

            plt.plot(epochs_range, self.history.history["loss"], label="Train Loss")
            plt.plot(epochs_range, self.history.history["val_loss"], label="Validation Loss")
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.title("Training vs Validation Loss")
            plt.xticks(epochs_range)
            plt.legend()
            plt.savefig("lstm_loss.png")
            plt.show()
        else:
            print("No training history found. Train the model first!")

    from sklearn.metrics import confusion_matrix, classification_report

    def plot_confusion_matrix(self):
        """Generate and plot the confusion matrix for LSTM predictions."""
        
        # Step 1: Get predictions (probabilities)
        y_pred_prob = self.model.predict(self.X_test)

        # Step 2: Convert probabilities to class labels (argmax to get highest probability index)
        y_pred = np.argmax(y_pred_prob, axis=1)

        # Step 3: Compute confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)

        # Step 4: Plot confusion matrix
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Negative", "Neutral", "Positive"], yticklabels=["Negative", "Neutral", "Positive"])
        
        plt.xlabel("Predicted Label", fontsize=12)
        plt.ylabel("True Label", fontsize=12)
        plt.title("LSTM Confusion Matrix", fontsize=14, fontweight="bold")
        plt.savefig("lstm_confusion_matrix.png")
        plt.show()

        # Print classification report
        print("\nClassification Report:\n")
        print(classification_report(self.y_test, y_pred, target_names=["Negative", "Neutral", "Positive"]))



