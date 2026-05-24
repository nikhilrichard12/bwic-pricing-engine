"""
Synthetic BWIC (Bid Wanted in Competition) loan list generator.
Produces messy_bwic_data.csv with 50 corporate loans and intentional data quality issues.
"""

import string

import numpy as np
import pandas as pd

#How many loan rows to emit in the BWIC tape
NUM_LOANS = 50

#Characters allowed in a CUSIP (letters + digits- real CUSIPs use a subset so simplified here)
CUSIP_CHARS = string.ascii_uppercase + string.digits

#Pool of issuer / facility labels used to build realistic Facility_Name values
ISSUER_NAMES = [
    "American Airlines",
    "Delta Air Lines",
    "United Airlines",
    "Spirit Airlines",
    "Carnival Cruise Lines",
    "Norwegian Cruise Line",
    "MGM Resorts",
    "Caesars Entertainment",
    "Hertz Global",
    "Party City",
    "J.Crew",
    "Revlon",
    "Neiman Marcus",
    "PetSmart",
    "Serta Simmons",
    "Claire's Stores",
    "iHeartMedia",
    "Tenet Healthcare",
    "Community Health Systems",
    "Tenet Physician Resources",
    "McClatchy",
    "Gannett",
    "Frontier Communications",
    "Windstream",
    "CenturyLink",
    "Dish Network",
    "DirecTV",
    "Charter Communications",
    "Cox Communications",
    "Altice USA",
]

# Loan tranche suffixes appended to issuer names for Facility_Name
FACILITY_SUFFIXES = [
    "Term Loan B",
    "Term Loan B-1",
    "Term Loan B-2",
    "Revolver",
    "First Lien Term Loan",
    "Second Lien Term Loan",
    "Incremental Term Loan",
    "DIP Term Loan",
]

# Moody’s-style speculative-grade buckets for this
CREDIT_RATINGS = ["B", "B-", "CCC+", "CCC"]

# Output path for the imperfect CSV
OUTPUT_CSV = "messy_bwic_data.csv"


def random_cusip(length: int = 9) -> str:
    """
    Build one random CUSIP-like string of exactly `length` characters.
    Each character is drawn uniformly from A–Z and 0–9.
    """
    # np.random.choice samples with replacement from CUSIP_CHARS, size=length times
    chars = np.random.choice(list(CUSIP_CHARS), size=length, replace=True)
    # Join the array of single-character strings into one 8- or 9-char CUSIP
    return "".join(chars)


def build_facility_name() -> str:
    """
    Compose a single Facility_Name by pairing a random issuer with a random tranche label.
    Example output: "American Airlines Term Loan B".
    """
    issuer = np.random.choice(ISSUER_NAMES)
    suffix = np.random.choice(FACILITY_SUFFIXES)
    return f"{issuer} {suffix}"


def generate_clean_loan_table(n: int) -> pd.DataFrame:
    """
    Create a DataFrame with `n` rows and clean synthetic loan fields.
    No missing values or short CUSIPs yet — mess is applied later.
    """
    #One CUSIP per row (all valid 9-character IDs at this stage)
    cusips = [random_cusip(9) for _ in range(n)]

    # One facility label per row
    facility_names = [build_facility_name() for _ in range(n)]

    # Par in dollars ints from 1_000_000 through 10_000_000 inclusive
    par_amounts = np.random.randint(1_000_000, 10_000_001, size=n)

    # One rating per row from the fixed four-rating universe
    credit_ratings = np.random.choice(CREDIT_RATINGS, size=n)

    # Price as percent of par: uniform floats in [85.0, 100.0], rounded to 2 decimals
    current_prices = np.round(np.random.uniform(85.0, 100.0, size=n), 2)

    # Assemble columns into a labeled table (pandas DataFrame)
    return pd.DataFrame(
        {
            "CUSIP": cusips,
            "Facility_Name": facility_names,
            "Par_Amount": par_amounts,
            "Credit_Rating": credit_ratings,
            "Current_Price": current_prices,
        }
    )


def introduce_messy_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mutate a copy of `df` to mimic real-world BWIC file problems:
    - 3 random Par_Amount cells set to missing (blank in CSV)
    - 2 random CUSIP values truncated to 8 characters
    Returns the modified DataFrame (original is unchanged).
    """
    messy = df.copy()

    # Pick 3 distinct row indices (no replacement) for blank par
    par_blank_indices = np.random.choice(messy.index, size=3, replace=False)
    # np.nan becomes an empty field when written with default pandas CSV rules
    messy.loc[par_blank_indices, "Par_Amount"] = np.nan

    # Pick 2 distinct rows for invalid short CUSIPs (avoid overlap not required)
    short_cusip_indices = np.random.choice(messy.index, size=2, replace=False)
    for idx in short_cusip_indices:
        # Replace the 9-char CUSIP with a fresh 8-char draw
        messy.at[idx, "CUSIP"] = random_cusip(8)

    return messy


def main() -> None:
    """Generate clean data, apply mess, and write messy_bwic_data.csv."""
    #1: build 50-row clean loan list
    loans = generate_clean_loan_table(NUM_LOANS)

    #2: inject blanks and short CUSIPs on a copy
    messy_loans = introduce_messy_data(loans)

    #3: persist; NaN Par_Amount cells export as empty strings in the CSV
    messy_loans.to_csv(OUTPUT_CSV, index=False)

    print(f"Wrote {len(messy_loans)} loans to {OUTPUT_CSV}")
    print(f"  Blank Par_Amount rows: {messy_loans['Par_Amount'].isna().sum()}")
    print(
        f"  Short CUSIPs (length != 9): "
        f"{(messy_loans['CUSIP'].str.len() != 9).sum()}"
    )


if __name__ == "__main__":
    main()
