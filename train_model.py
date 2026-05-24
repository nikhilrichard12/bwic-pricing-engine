"""
Train a Random Forest on synthetic historical loan trades and export artifacts for the app.
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

NUM_ROWS = 5_000
RANDOM_STATE = 42

CREDIT_RATINGS = ["B", "B-", "CCC+", "CCC"]
SECTORS = ["Tech", "Healthcare", "Industrials", "Consumer"]

# Anchor execution levels by rating before volatility/SOFR adjustments
RATING_PRICE_ANCHOR = {"B": 99.0, "B-": 96.5, "CCC+": 93.0, "CCC": 88.0}

MODEL_PATH = "pricing_model.joblib"
FEATURES_PATH = "model_features.joblib"

CATEGORICAL_COLS = ["Credit_Rating", "Sector"]
NUMERIC_FEATURE_COLS = ["Par_Amount", "Market_Volatility", "SOFR_Rate"]
TARGET_COL = "Execution_Price"


def generate_historical_trades(n: int) -> pd.DataFrame:
    """Build synthetic trade-level history with loan, market, and macro fields."""
    rng = np.random.default_rng(RANDOM_STATE)

    return pd.DataFrame(
        {
            "Par_Amount": rng.integers(1_000_000, 10_000_001, size=n),
            "Credit_Rating": rng.choice(CREDIT_RATINGS, size=n),
            "Sector": rng.choice(SECTORS, size=n),
            "Market_Volatility": np.round(rng.uniform(10.0, 30.0, size=n), 2),
            "SOFR_Rate": np.round(rng.uniform(4.0, 5.5, size=n), 2),
        }
    )


def build_execution_price(df: pd.DataFrame) -> pd.Series:
    """
    Synthetic target: better ratings and calmer markets print closer to par;
    wider vol and higher SOFR drag price down, plus Gaussian noise.
    """
    rng = np.random.default_rng(RANDOM_STATE + 1)

    base = df["Credit_Rating"].map(RATING_PRICE_ANCHOR)
    vol_drag = (df["Market_Volatility"] - 10.0) * 0.12
    sofr_drag = (df["SOFR_Rate"] - 4.0) * 0.8
    noise = rng.normal(0, 1.2, size=len(df))

    price = base - vol_drag - sofr_drag + noise
    return pd.Series(np.clip(price, 70.0, 100.0), index=df.index, name=TARGET_COL)


def preprocess_features(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode categoricals; numeric features pass through unchanged."""
    feature_df = df[NUMERIC_FEATURE_COLS + CATEGORICAL_COLS].copy()
    return pd.get_dummies(feature_df, columns=CATEGORICAL_COLS, dtype=int)


def main() -> None:
    np.random.seed(RANDOM_STATE)

    trades = generate_historical_trades(NUM_ROWS)
    trades[TARGET_COL] = build_execution_price(trades)

    X = preprocess_features(trades)
    y = trades[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=12,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Test set Mean Absolute Error (MAE): {mae:.4f}")

    joblib.dump(model, MODEL_PATH)
    joblib.dump(X.columns.tolist(), FEATURES_PATH)
    print(f"Saved model -> {MODEL_PATH}")
    print(f"Saved feature names ({len(X.columns)} columns) -> {FEATURES_PATH}")


if __name__ == "__main__":
    main()
