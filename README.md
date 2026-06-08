# Predictive Maintenance ML Platform
### Real-Time Machine Failure Prediction | Random Forest | Live on AWS EC2

---

## What Business Problem Does This Solve?

An unplanned machine breakdown on a factory floor costs an average of
**₹5–20 lakhs per hour** in downtime, lost production, and emergency repairs.
Traditional maintenance is either reactive (fix after failure) or scheduled
(wasteful — servicing machines that don't need it).

**This project answers the question every operations manager needs answered:**
> *"Will this machine fail in the next cycle — before it actually does?"*

---

## Business Results

| Metric | Value |
|---|---|
| Model accuracy | Random Forest classifier |
| Input signals | Air temp, process temp, rotational speed, torque, tool wear |
| Prediction | Binary — failure / no failure in real-time |
| Deployment | Live on AWS EC2 — accessible via browser |
| Response time | Instant inference from IoT sensor inputs |

---

## Key Business Impact

- **Prevents unplanned downtime** — maintenance teams get early warning before failure occurs
- **Reduces maintenance cost** — service only machines that show failure signals, not all machines on schedule
- **Works in real-time** — operations staff enter live sensor readings and get instant prediction
- **No data science knowledge needed** — simple Streamlit UI built for floor managers, not analysts

---

## Live Demo

🌐 **[Live App on AWS EC2](http://43.205.113.174:8502)**

Enter any sensor values and get an instant failure prediction.

---

## How It Works

```
IoT Sensor Readings (live input)
    │  Air Temp · Process Temp · RPM · Torque · Tool Wear
    ▼
Streamlit Web App (browser-based UI)
    │
    ▼
Random Forest Classifier (.pkl model)
    │  Trained on 10,000 labelled machine records
    ▼
Prediction Output
    ├── ✅ No Failure Expected
    └── ⚠️  Failure Risk Detected → Alert maintenance team
```

---

## ML Model Details

| Parameter | Detail |
|---|---|
| Algorithm | Random Forest Classifier |
| Library | Scikit-Learn |
| Features | 5 sensor inputs (temperature, speed, torque, wear) |
| Target | Binary classification (failure / no failure) |
| Dataset | UCI AI4I 2020 Predictive Maintenance Dataset |

---

## Deployment Architecture

```
AWS EC2 (t2.medium)
└── Docker Container
    ├── Streamlit app.py    → Port 8502
    └── model.pkl           → Random Forest (pre-trained)
```

Deployed as a Docker container on AWS EC2. No database required —
model is loaded from a `.pkl` file at startup. Zero external dependencies.

---

## Tech Stack

| Layer | Tool |
|---|---|
| ML model | Python, Scikit-Learn, Random Forest |
| Web app | Streamlit |
| Deployment | Docker, AWS EC2 |
| Dataset | UCI AI4I 2020 (10,000 records, 5 features) |

---

## How to Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/premchavan772005-spec/Predictive-Maintenance-ML-Pipeline

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

---

*Built by Prem Chavan | Data Analyst*
*Skills: Python · Scikit-Learn · Machine Learning · Streamlit · Docker · AWS EC2*
