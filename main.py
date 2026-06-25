from rag import load_data

if __name__ == "__main__":
    merged, news_df = load_data(
        r"C:\Users\sanas\Downloads\archive\news_sentiment_raw.csv",
        r"C:\Users\sanas\Downloads\archive\market_prices.csv"
    )
    print(f"Daily panel: {merged.shape}")
    print(f"News corpus: {len(news_df)} articles across {news_df['date'].nunique()} dates")
    print(merged.head(3))