# Predictive Maintenance End-to-End ML & Analytics Pipeline
🚀 **[👉 Live ML Demo — Click Here](https://predictive-maintenance-ml-pipeline-keqeaamyyt8glyzdw4jype.streamlit.app/)**

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

   ## 📊 Executive BI Dashboard Interface

### Page 1: Operational Factory Control Center
*Provides an immediate high-level macro overview of factory floor uptime alongside an unaggregated micro-breakdown of mechanical failure bottlenecks.*

✨ ![Page 1 - Operational Factory Control Center](maintenance%20page1.png)

### Page 2: Predictive Maintenance Projections
*Integrates the live Scikit-Learn Random Forest inference data, mapping anomaly clusters and serving an urgent high-risk maintenance action checklist sorted by machine failure probability.*

✨ ![Page 2 - Predictive Maintenance Projections](maintenance%20page2.png)
