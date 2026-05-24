"""
BWIC pricing engine — upload a loan tape CSV, inspect raw rows, and view cleaned data.
"""

import pandas as pd
import streamlit as st


def clean_bwic_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows with missing par and invalid (non-9-char) CUSIPs.
    Returns a new DataFrame; does not mutate the input.
    """
    cleaned = df.dropna(subset=["Par_Amount"])
    cusip_as_str = cleaned["CUSIP"].astype(str)
    cleaned = cleaned[cusip_as_str.str.len() == 9].copy()
    return cleaned


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
        st.metric("Rows dropped (bad data)", rows_dropped)
        st.dataframe(cleaned_df, use_container_width=True)
