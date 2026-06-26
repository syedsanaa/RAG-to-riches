from rag import load_data, topic_sentiment_features, train_test_split_by_date, train_model, evaluate_model

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

    X_train, X_test, y_train, y_test = train_test_split_by_date(feature_df)
    model = train_model(X_train, y_train)
    results = evaluate_model(model, X_test, y_test)
    print("V1 results:", results)