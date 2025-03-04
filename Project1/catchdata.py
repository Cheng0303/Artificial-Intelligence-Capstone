import praw
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Reddit API Authentication
reddit = praw.Reddit(
    client_id="-Lf81RKaoxrATcdUtYPDaw",
    client_secret="hHd8uhSG_Dqz6zS8gWr6dhhTrVJ0-Q",
    user_agent="python:RedditScraper:1.0 (by u/BookSignificant9843)"
)

# Choose a subreddit and collect posts
subreddits = ["ArtificialInteligence", "StockMarket", "Music", "gaming", "business", "Scams", "jobs", 'community', 'nba', 'movies', 'education', 'comicbooks', 'Meditation']
posts = []

for sub in subreddits:
    subreddit = reddit.subreddit(sub)
    for post in subreddit.hot(limit=100):  # Adjust limit as needed
        posts.append([post.title, post.selftext, post.score, post.num_comments, post.created_utc])

# Convert to DataFrame
df = pd.DataFrame(posts, columns=["Title", "Body", "Score", "Comments", "Timestamp"])
df = df[df["Body"].str.strip().astype(bool)]  # Remove empty posts
df.to_csv("dataset/reddit_data100.csv", index=False)
print("Scraping complete. Data saved!")

analyzer = SentimentIntensityAnalyzer()

def get_sentiment(text):
    score = analyzer.polarity_scores(text)
    return score["compound"]  # -1 (negative) to +1 (positive)

df["Sentiment"] = df["Body"].apply(get_sentiment)
df["Sentiment_Label"] = df["Sentiment"].apply(lambda x: "Positive" if x > 0.05 else "Negative" if x < -0.05 else "Neutral")
df.to_csv("dataset/reddit_sentiment100.csv", index=False)
print("Sentiment analysis complete. Data saved!")

