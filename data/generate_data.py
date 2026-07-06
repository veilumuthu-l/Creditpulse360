"""
Synthetic MSME dataset generator for CreditPulse360.

Simulates the *shape* of data CreditPulse360 would receive in production from:
  - GSTN (via a licensed GSP)             -> gst_* features
  - Bank statements fetched via an        -> upi_* features (UPI activity as it
    Account Aggregator (post UPI-on-AA)      appears inside bank statements)
  - Bank statements via Account Aggregator -> aa_* features

This is clearly-labeled SYNTHETIC data for prototype/demo purposes only.
It is NOT real GST/UPI/bank data and must be replaced with sandboxed,
consented real data during the pilot phase (see README.md roadmap).
"""

import numpy as np
import pandas as pd
from faker import Faker

fake = Faker("en_IN")
rng = np.random.default_rng(42)

N = 1200  # number of synthetic MSME applicants

SECTORS = ["Retail/Kirana", "Manufacturing", "Agri-trade", "Auto components",
           "Textiles", "Food & Beverage", "Services", "Wholesale Trade"]


def generate_dataset(n=N):
    rows = []
    for i in range(n):
        sector = rng.choice(SECTORS)
        vintage_months = int(rng.integers(4, 96))  # business age

        # --- GST-derived features ---
        gst_filing_regularity = np.clip(rng.beta(5, 2), 0, 1)          # 0-1, higher = more regular
        gst_turnover_growth = rng.normal(0.06, 0.15)                    # YoY growth, can be negative
        gst_itc_mismatch_rate = np.clip(rng.beta(2, 8), 0, 1)           # input-output tax mismatch, lower is better

        # --- UPI-derived features (from bank statement analysis) ---
        upi_avg_daily_inflow = max(500, rng.normal(18000, 12000))
        upi_inflow_volatility = np.clip(rng.beta(2, 5), 0, 1)           # coefficient of variation, lower is better
        upi_customer_concentration = np.clip(rng.beta(2, 6), 0, 1)      # % revenue from top-3 payers, lower is better

        # --- Account Aggregator bank-behaviour features ---
        aa_avg_balance = max(0, rng.normal(45000, 30000))
        aa_bounce_rate = np.clip(rng.beta(1.5, 10), 0, 1)               # cheque/mandate bounce rate, lower is better
        aa_existing_emi_ratio = np.clip(rng.beta(2, 6), 0, 1)           # existing EMI / inflow, lower is better

        # --- thin bureau / vintage signal ---
        bureau_thin_signal = np.clip(rng.beta(3, 3), 0, 1)              # proxy for any partial bureau footprint

        rows.append(dict(
            applicant_id=f"MSME{1000+i}",
            business_name=fake.company(),
            sector=sector,
            vintage_months=vintage_months,
            gst_filing_regularity=round(gst_filing_regularity, 3),
            gst_turnover_growth=round(gst_turnover_growth, 3),
            gst_itc_mismatch_rate=round(gst_itc_mismatch_rate, 3),
            upi_avg_daily_inflow=round(upi_avg_daily_inflow, 2),
            upi_inflow_volatility=round(upi_inflow_volatility, 3),
            upi_customer_concentration=round(upi_customer_concentration, 3),
            aa_avg_balance=round(aa_avg_balance, 2),
            aa_bounce_rate=round(aa_bounce_rate, 3),
            aa_existing_emi_ratio=round(aa_existing_emi_ratio, 3),
            bureau_thin_signal=round(bureau_thin_signal, 3),
        ))

    df = pd.DataFrame(rows)

    # --- Ground-truth default risk formula (used ONLY to simulate labels) ---
    # Higher score = lower risk. Weights loosely mirror the "scoring methodology" slide.
    risk_score = (
        -1.8 * df.gst_filing_regularity
        - 1.0 * df.gst_turnover_growth
        + 1.4 * df.gst_itc_mismatch_rate
        - 0.9 * (df.upi_avg_daily_inflow / df.upi_avg_daily_inflow.max())
        + 1.6 * df.upi_inflow_volatility
        + 1.1 * df.upi_customer_concentration
        - 0.8 * (df.aa_avg_balance / df.aa_avg_balance.max())
        + 2.0 * df.aa_bounce_rate
        + 1.3 * df.aa_existing_emi_ratio
        - 0.5 * df.bureau_thin_signal
        - 0.4 * (df.vintage_months / df.vintage_months.max())
    )
    prob_default = 1 / (1 + np.exp(-(risk_score + rng.normal(0, 0.5, n))))  # add noise
    df["default_12m"] = (rng.random(n) < prob_default).astype(int)

    return df


if __name__ == "__main__":
    df = generate_dataset()
    df.to_csv("data/msme_synthetic_dataset.csv", index=False)
    print(f"Generated {len(df)} synthetic MSME records -> data/msme_synthetic_dataset.csv")
    print(f"Default rate in synthetic data: {df.default_12m.mean():.1%}")
