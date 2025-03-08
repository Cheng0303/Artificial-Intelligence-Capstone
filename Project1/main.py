import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sentence_transformers import SentenceTransformer
from Randomforest import RandomForest
from LSTM import LSTMModel
from Kmean import KMean
from imblearn.over_sampling import SMOTE

bigFileName = "dataset/reddit_sentiment1000.csv"
smallFileName = "dataset/reddit_sentiment100.csv"
df = pd.read_csv(bigFileName)
df = df.dropna(subset=["Body"])

# # Convert text into numerical features
# bert_model = SentenceTransformer("all-MiniLM-L6-v2")

# # Convert text to BERT embeddings
# X = bert_model.encode(df["Body"].tolist())

vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df["Body"])
y = df["Sentiment_Label"]

# Apply SMOTE for Class Imbalance
X, y = SMOTE(random_state=42).fit_resample(X, y)

# Train/Validation/Test Split (70% / 15% / 15%)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)


# Train Random Forest Model For bigger dataset
print("====================Random Forest====================")
RF = RandomForest()
RF.fit(X_train, y_train)
RF.predict(X_test, y_test)

# Train KMean Model
print("====================KMean====================")
kmean = KMean(num_clusters=3)
kmean.fit(X, y)

# Train LSTM Model
print("====================LSTM====================")
lstm = LSTMModel(df)
lstm.train()
lstm.predict()
lstm.plot_loss()
lstm.plot_confusion_matrix()


# Train Random Forest Model For smaller dataset
print("====================Random Forest(small)====================")
sdf = pd.read_csv(smallFileName)
sdf = sdf.dropna(subset=["Body"])
sX = vectorizer.fit_transform(sdf["Body"])
sy = sdf["Sentiment_Label"]

RF = RandomForest()
RF.fit(sX, sy)
RF.predict(X_test, y_test)

# Train LSTM Model For smaller dataset
print("====================LSTM(small)====================")
X = df.drop(columns=["Sentiment_Label"])
y = df["Sentiment_Label"].map({"Negative": 0, "Neutral": 1, "Positive": 2})
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print(y_test.name)
lstm = LSTMModel(sdf)
lstm.train()
lstm.predict(X_test, y_test)
lstm.plot_loss()
lstm.plot_confusion_matrix()

