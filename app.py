"""
BWIC pricing engine — upload a loan tape CSV, inspect raw rows, and view cleaned data.
"""

import pandas as pd
import streamlit as st

# Moody's Weighted Average Rating Factor by letter rating (speculative-grade tape)
MOODYS_WARF: dict[str, int] = {
    "B": 2830,
    "B-": 3490,
    "CCC+": 4770,
    "CCC": 6500,
}


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


st.set_page_config(page_title="BWIC Pricing Engine", layout="wide")
st.title("BWIC Pricing Engine")

uploaded_file = st.file_uploader("Upload a BWIC loan tape (CSV)", type=["csv"])

if uploaded_file is None:
    st.info("Upload a CSV file to view raw and cleaned loan data.")
else:
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

        st.dataframe(cleaned_df, use_container_width=True)
