from pathlib import Path
import pandas as pd

def load_raw_data(raw_path: Path) -> pd.DataFrame:
    df = pd.read_csv("C:/Users/thota/OneDrive/Desktop/sales-time-series-forecasting/data/raw/train.csv")
    return df

def clean_dates(df: pd.DataFrame, date_col: str = "Order Date") -> pd.DataFrame:
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], dayfirst=True, errors="coerce")
    df = df.dropna(subset=[date_col])
    return df

def aggregate_monthly_sales(df: pd.DataFrame, date_col: str = "Order Date", sales_col: str = "Sales") -> pd.DataFrame:
    monthly = (
        df.groupby(pd.Grouper(key=date_col, freq="MS"))[sales_col]
          .sum()
          .sort_index()
          .reset_index()
    )
    monthly.columns = ["date", "sales"]
    return monthly

def main():
    raw_path = Path("data/raw/train.csv")
    out_path = Path("data/processed/monthly_sales.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not raw_path.exists():
        raise FileNotFoundError(
            f"Raw data not found at {raw_path}. "
            "Put train.csv into data/raw/train.csv"
        )

    df = load_raw_data(raw_path)
    df = clean_dates(df, date_col="Order Date")
    monthly_sales = aggregate_monthly_sales(df, date_col="Order Date", sales_col="Sales")

    monthly_sales.to_csv(out_path, index=False)
    print(f"Saved monthly sales to: {out_path}")
    print(monthly_sales.head())

if __name__ == "__main__":
    main()