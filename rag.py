
import pandas as pd

def load_data(news_path: str, market_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
     1)We are merging the news and market data based on date because 
    we want to see for a particular date based on the sentiment 
    today what was the effect on Market tomorrow.
    2)We are assuming that for a date there are multiple rows in 
    the news data and one row in the markets data
    Returns:
        merged_df  — news rows with market data broadcast across (N:1)
        market_df  — one row per date, for use in aggregate_by_date
    """
    news_df = pd.read_csv(news_path, parse_dates=["date"])
    market_df = pd.read_csv(market_path, parse_dates=["date"])

    news_df["date"] = news_df["date"].dt.normalize()
    market_df["date"] = market_df["date"].dt.normalize()
    market_df = market_df.sort_values("date")
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

