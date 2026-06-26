from rag import load_data, topic_sentiment_features

if __name__ == "__main__":
    merged, news_df = load_data(
        r"C:\Users\sanas\Downloads\archive\news_sentiment_raw.csv",
        r"C:\Users\sanas\Downloads\archive\market_prices.csv"
    )
    print(f"Daily panel: {merged.shape}")
    print(f"News corpus: {len(news_df)} articles across {news_df['date'].nunique()} dates")
    print(merged.head(3))

    feature_df = topic_sentiment_features(news_df, merged)
    print(f"Feature matrix: {feature_df.shape}")
    print(feature_df.head(3))