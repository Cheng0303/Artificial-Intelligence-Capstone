import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_csv("dataset/reddit_sentiment1000.csv")
df = df.dropna(subset=["Body"])

# Example Data
df["Sentiment_Label"] = df["Sentiment_Label"].map({"Negative": 0, "Neutral": 1, "Positive": 2})

# Tokenize Text
max_words = 5000
max_length = 50

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(df["Body"])
X = tokenizer.texts_to_sequences(df["Body"])
X = pad_sequences(X, maxlen=max_length)
y = df["Sentiment_Label"].values

# Train/Validation/Test Split
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Build LSTM Model
model = Sequential([
    Embedding(input_dim=max_words, output_dim=128, input_length=max_length),
    LSTM(64, return_sequences=True),  # This should work now
    LSTM(32),
    Dropout(0.5),
    Dense(3, activation="softmax")  # Multi-class classification
])

# Compile Model
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train Model
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=5, batch_size=32)

# Evaluate
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")
