# CreditPulse360

**AI-powered Financial Health Score for credit-invisible, New-to-Bank MSMEs.**

Submission for IDBI Innovate 2026 — Track 03: Financial Health Score.

> **Prototype scope & honesty note:** This is a working, end-to-end prototype
> trained on **synthetic data** (see `data/generate_data.py`) that mimics the
> *shape* of GST, UPI, and Account Aggregator data. It demonstrates the full
> scoring pipeline — feature engineering, ML model, and real SHAP-based
> explainability — but is **not connected to live GSTN/UPI/AA systems**.
> That integration is the first step of the production roadmap below.

---

## What's actually real here

- A real trained XGBoost model (not a mock) predicting default risk
- Real SHAP explainability — every reason code shown in the app is computed
  live from the model for that specific applicant, not hardcoded text
- A working interactive dashboard (portfolio view + applicant explorer)
- Synthetic data standing in for GST/UPI/AA feeds
- Simulated default labels (no real repayment outcomes exist yet)

## Quick start (local)

```bash
git clone <your-repo-url>
cd creditpulse360
pip install -r requirements.txt

python data/generate_data.py   # generates synthetic dataset
python model/train_model.py    # trains model + SHAP explainer

streamlit run app.py
```

Open the URL Streamlit prints (usually `http://localhost:8501`).

## Deploy a public demo in minutes (Streamlit Community Cloud — free)

1. Push this folder to a **public GitHub repo**.
2. Commit the generated files too (`data/msme_synthetic_dataset.csv` and
   `model/creditpulse_model.pkl`) so the app doesn't need to retrain on boot —
   or leave a `prestart` step; simplest is to commit them.
3. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub.
4. Click **New app** → select your repo → main file path `app.py` → **Deploy**.
5. You'll get a public URL like `https://your-app.streamlit.app` in ~2 minutes.

This is the fastest path to a live, judge-clickable demo link for your submission.

### Alternative: Render.com (also free tier)
If you'd rather not commit the `.pkl`/`.csv` artifacts, add a build command:
`pip install -r requirements.txt && python data/generate_data.py && python model/train_model.py`
and start command: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`

## Project structure

```
creditpulse360/
├── data/
│   └── generate_data.py       # synthetic MSME data generator
├── model/
│   └── train_model.py         # trains XGBoost + saves SHAP explainer
├── app.py                     # Streamlit dashboard (portfolio + applicant view)
├── requirements.txt
└── README.md
```

## Scoring methodology

Financial Health Score (300–900) = `900 - P(default_12m) * 600`, where
`P(default_12m)` comes from an XGBoost classifier trained on:

| Signal source | Features |
|---|---|
| GST (via GSP) | filing regularity, turnover growth, input-tax mismatch rate |
| UPI (via bank statement/AA) | avg. daily inflow, volatility, customer concentration |
| Account Aggregator | avg. balance, bounce rate, existing EMI ratio |
| Other | bureau thin-file signal, business vintage |

## Roadmap: prototype → production

| Phase | What changes |
|---|---|
| **This repo (Phase 0)** | Synthetic data, demo-ready, explainability proven |
| **Phase 1 — Sandbox integration** | Replace `generate_data.py` with real GSP/AA sandbox pulls; re-validate model on real-shaped anonymized data |
| **Phase 2 — Compliance** | RBI Digital Lending Guidelines review, DPDP Act consent audit, model risk sign-off |
| **Phase 3 — Shadow-mode pilot** | Score real historical applicants, compare vs. actual underwriting outcomes, collect real default labels |
| **Phase 4 — LOS integration** | Deploy scoring API as a microservice behind IDBI's Loan Origination System, advisory mode first |
| **Phase 5 — Auto-decisioning** | Only after pilot validates accuracy; human-in-the-loop for medium-risk band retained |

## License / disclaimer

Built for hackathon demonstration purposes. All data is synthetic. Not
financial advice, not a production lending system.
