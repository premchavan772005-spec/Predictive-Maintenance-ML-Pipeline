# Predictive Maintenance End-to-End ML & Analytics Pipeline

## 📈 System Architecture Overview
This production-grade platform ingests real-time industrial IoT telemetry, streams records into a central relational data warehouse, trains an ensemble Random Forest model to predict mechanical anomalies, and serves interactive risk projections on an executive Power BI dashboard layer.
## 🛠️ Tech Stack & Core Infrastructure
* **Data Warehouse:** MySQL Engine (Relational Database)
* **ETL Engine / Orchestration:** Python 3.14 (Pandas, SQLAlchemy, PyMySQL)
* **Predictive ML Core:** Scikit-Learn (Random Forest Ensemble Classifier)
* **BI Presentation Layer:** Power BI Desktop (Dual-Page Executive Dashboards)

## 📊 Business Key Performance Indicators (KPIs)
* **Total Ingested Data Streams:** 10,000 baseline telemetry tracking logs
* **Total Captured Anomaly Events:** 339 confirmed failure profiles
* **ML Precision Target:** 89% (Extremely low false-alarm rate for floor crews)
* **ML Robust F1-Score:** 70% (Optimized benchmark for highly imbalanced anomaly datasets)

## 🚀 How to Run the Pipeline
1. Clone this repository to your local runtime space.
2. Spin up the local warehouse schema using the queries in `maintenance_warehouse.sql`.
3. Execute the ETL automated data stream pipeline script:
   ```bash
   python maintenance_etl.py