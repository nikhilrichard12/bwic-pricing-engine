"""
BWIC pricing engine — upload a loan tape CSV, inspect raw rows, and view cleaned data.
"""

import joblib
import pandas as pd
import streamlit as st

# Moody's Weighted Average Rating Factor by letter rating (speculative-grade tape)
MOODYS_WARF: dict[str, int] = {
    "B": 2830,
    "B-": 3490,
    "CCC+": 4770,
    "CCC": 6500,
}

MODEL_PATH = "pricing_model.joblib"
FEATURES_PATH = "model_features.joblib"
CATEGORICAL_COLS = ["Credit_Rating", "Sector"]
NUMERIC_FEATURE_COLS = ["Par_Amount", "Market_Volatility", "SOFR_Rate"]


@st.cache_resource
def load_model_artifacts() -> tuple[object, list[str]]:
    """Load persisted model and feature column list once per Streamlit session."""
    model = joblib.load(MODEL_PATH)
    expected_features = joblib.load(FEATURES_PATH)
    return model, expected_features


def clean_bwic_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows with missing par and invalid (non-9-char) CUSIPs.
    Returns a new DataFrame; does not mutate the input.
    """
    cleaned = df.dropna(subset=["Par_Amount"])
    cusip_as_str = cleaned["CUSIP"].astype(str)
    cleaned = cleaned[cusip_as_str.str.len() == 9].copy()
    return cleaned


def calculate_portfolio_metrics(df: pd.DataFrame) -> tuple[float, float]:
    """
    Par-weighted WAP and WARF for a cleaned loan tape.
    Adds a temporary WARF_Factor column via rating lookup.
    """
    work = df.copy()
    work["WARF_Factor"] = work["Credit_Rating"].map(MOODYS_WARF)

    total_par = work["Par_Amount"].sum()
    wap = (work["Par_Amount"] * work["Current_Price"]).sum() / total_par
    warf = (work["Par_Amount"] * work["WARF_Factor"]).sum() / total_par

    return round(wap, 2), round(warf, 2)


def predict_prices(
    df: pd.DataFrame, model: object, expected_features: list[str]
) -> pd.DataFrame:
    """
    One-hot encode categoricals, align to training columns, and append Predicted_Price.
    """
    result = df.copy()
    feature_cols = [c for c in NUMERIC_FEATURE_COLS + CATEGORICAL_COLS if c in df.columns]
    feature_df = df[feature_cols].copy()

    encode_cols = [c for c in CATEGORICAL_COLS if c in feature_df.columns]
    encoded = pd.get_dummies(feature_df, columns=encode_cols, dtype=int)

    model_input = encoded.reindex(columns=expected_features, fill_value=0)
    result["Predicted_Price"] = model.predict(model_input)

    return result


st.set_page_config(page_title="BWIC Pricing Engine", layout="wide")
st.title("BWIC Pricing Engine")

uploaded_file = st.file_uploader("Upload a BWIC loan tape (CSV)", type=["csv"])

if uploaded_file is None:
    st.info("Upload a CSV file to view raw and cleaned loan data.")
else:
    model, expected_features = load_model_artifacts()

    raw_df = pd.read_csv(uploaded_file)
    cleaned_df = clean_bwic_data(raw_df)
    rows_dropped = len(raw_df) - len(cleaned_df)

    tab_raw, tab_clean = st.tabs(["Raw Data", "Cleaned Data"])

    with tab_raw:
        st.dataframe(raw_df, use_container_width=True)

    with tab_clean:
        wap, warf = calculate_portfolio_metrics(cleaned_df)

        col_dropped, col_wap, col_warf = st.columns(3)
        with col_dropped:
            st.metric("Rows dropped (bad data)", rows_dropped)
        with col_wap:
            st.metric("Weighted Average Price (WAP)", wap)
        with col_warf:
            st.metric("Weighted Average Rating Factor (WARF)", warf)

        priced_df = predict_prices(cleaned_df, model, expected_features)
        st.dataframe(priced_df, use_container_width=True)
