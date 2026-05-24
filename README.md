# BWIC Pricing Engine
[Live App](https://bwic-pricing-engine-n12khil.streamlit.app/)

## The Problem
Syndicated loans and CLOs trade over-the-counter. When an asset manager wants to sell a portfolio of loans, they initiate a BWIC (Bids Wanted In Competition). These lists are sent to dealers as messy Excel files or unstructured text. Traders are forced to manually clean the data, calculate the portfolio risk, and formulate a competitive bid before a strict deadline. It is a slow, inefficient process.

## The Solution
I built this engine to automate that workflow. You upload a raw BWIC tape into the app, and the pipeline does three things:
1. **Data Cleaning:** Automatically catches and drops rows with missing par amounts or broken CUSIPs.
2. **Portfolio Analytics:** Calculates the Weighted Average Price (WAP) and the Weighted Average Rating Factor (WARF) to give an immediate read on aggregate risk.
3. **Predictive Pricing:** Runs the cleaned tape through a trained Random Forest model to predict the final clearing price for every single loan on the list.

## Tech Stack
* **Frontend:** Streamlit
* **Data Pipeline:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn (Random Forest Regressor)

## Run It Locally
If you want to pull this down and run it on your own machine:

```bash
# Clone the repo
git clone [https://github.com/nikhilrichard12/bwic-pricing-engine.git](https://github.com/nikhilrichard12/bwic-pricing-engine.git)
cd bwic-pricing-engine

# Set up the virtual environment
python3 -m venv venv
source venv/bin/activate

# Install the required packages
pip install -r requirements.txt

# Boot the local server
streamlit run app.py