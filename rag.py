
import pandas as pd

def load_data(news_path: str, market_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
     1)We are merging the news and market data based on date because 
    we want to see how sentiment affects market prices on daily basis
    2)We are assuming that for a date there are multiple rows in 
    the news data and one row in the markets data
    Returns:
        merged_df  — for one date we have averaged sentiment values 
        news_df  — returning this to preserve all the news data for further analysis
    """
    news_df = pd.read_csv(news_path, parse_dates=["date"])
    market_df = pd.read_csv(market_path, parse_dates=["date"])

    news_df["date"] = news_df["date"].dt.normalize()
    market_df["date"] = market_df["date"].dt.normalize()
    market_df = market_df.sort_values("date")
    # So we shift the date by one because for a date we have sentiment data and we want to see how it affects market prices next day
    market_df["spy_up_next1d"] = (market_df["spy_return_1d"].shift(-1) > 0).astype(int)
    assert market_df["date"].is_unique, (
        f"market_prices.csv has duplicate dates: "
        f"{market_df[market_df['date'].duplicated()]['date'].tolist()}"
    )
    daily_sentiment = news_df.groupby('date')['compound'].mean().reset_index()

    merged = pd.merge(daily_sentiment, market_df, on="date", how="inner")

    required_cols = { "compound", "date", "spy_up_next1d"}
    missing = required_cols - set(merged.columns)
    if missing:
        raise ValueError(f"Merged DataFrame is missing expected columns: {missing}")

    return merged, news_df

 
def topic_sentiment_features(news_df: pd.DataFrame, merged: pd.DataFrame) -> pd.DataFrame:
    """
    For each date, compute average sentiment per topic.
    Returns one row per date with columns: one per topic + spy_up_next1d as target.
    
    Why: instead of one avg sentiment signal per day, we get 7 (one per topic).
    This lets logistic regression tell us WHICH topic predicts market direction best.
    """
    # average compound score per date per topic
    topic_daily = (
        news_df.groupby(["date", "topic"])["compound"]
        .mean()
        .unstack(fill_value=0)  # topics become columns, missing = 0
    )
    topic_daily.columns = [f"sent_{col}" for col in topic_daily.columns]
    topic_daily = topic_daily.reset_index()

    # merge with merged (which has spy_up_next1d) on date
    feature_df = pd.merge(topic_daily, merged[["date", "spy_up_next1d"]], on="date", how="inner")

    return feature_df

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score

def train_test_split_by_date(feature_df: pd.DataFrame, cutoff: str = "2023-01-01"):
    feature_df = feature_df.sort_values("date").reset_index(drop=True)
    split_idx = int(len(feature_df) * 0.8)
    
    train = feature_df.iloc[:split_idx]
    test = feature_df.iloc[split_idx:]

    feature_cols = [col for col in feature_df.columns if col.startswith("sent_")]

    X_train = train[feature_cols]
    X_test = test[feature_cols]
    y_train = train["spy_up_next1d"]
    y_test = test["spy_up_next1d"]

    print(f"Train: {len(train)} days | Test: {len(test)} days")
    return X_train, X_test, y_train, y_test


def train_model(X_train: pd.DataFrame, y_train: pd.Series):
    """Logistic regression — chosen for interpretability not just accuracy. We have less data so any other complex model will cause overfitting """
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Evaluate on held-out test set only — never on training data."""
    preds = model.predict(X_test)
    results = {
        "accuracy": round(accuracy_score(y_test, preds), 3),
        "precision": round(precision_score(y_test, preds, zero_division=0), 3),
        "recall": round(recall_score(y_test, preds, zero_division=0), 3)
    }
    return results