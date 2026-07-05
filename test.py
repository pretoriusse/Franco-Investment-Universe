import yfinance as yf


def fetch_and_show(ticker: str):
    # Download full daily history
    df = yf.download(
        ticker,
        period="max",
        interval="1d",
        auto_adjust=False,  # set to True if you want adjusted Close in 'Close'
        progress=False,
    )

    # Reset index so 'Date' becomes a column
    df.reset_index(inplace=True)

    # Show first / last 5 rows
    print(f"=== Head for {ticker} ===")
    print(df.head(), "\n")
    print(f"=== Tail for {ticker} ===")
    print(df.tail())


if __name__ == "__main__":
    fetch_and_show("AVI.JO")
