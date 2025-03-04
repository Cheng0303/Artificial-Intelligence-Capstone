import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from Randomforest import RandomForest
from Kmean import KMean

bigFileName = "dataset/reddit_sentiment1000.csv"
smallFileName = "dataset/reddit_sentiment100.csv"
df = pd.read_csv(bigFileName)

# Convert text into numerical features
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df["Body"])
y = df["Sentiment_Label"]

# Train/Validation/Test Split (70% / 15% / 15%)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Train Random Forest Model For bigger dataset
RF = RandomForest()
RF.fit(X_train, y_train)
RF.predict(X_test, y_test)

# Train KMean Model
kmean = KMean(num_clusters=3)
df["Cluster"] = kmean.fit(X_train)

