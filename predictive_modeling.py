import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# --- WAREHOUSE CONNECTOR CONFIGURATION ---
DB_USER = "root"
DB_PASSWORD = "its_prem7725$67"  # ⚠️ Change this to your actual password
DB_HOST = "localhost"
DB_PORT = "3306"
DB_NAME = "predictive_maintenance_db"

def train_failure_predictor():
    print("🔌 [1/5] Extracting live telemetry logs directly from MySQL Warehouse...")
    try:
        connection_string = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
        engine = create_engine(connection_string)
        
        # Read the fact table directly into a pandas dataframe
        query = "SELECT * FROM fact_sensor_logs;"
        df = pd.read_sql(query, con=engine)
        print(f" Data fetched. Loaded {df.shape[0]} tracking logs for model ingestion.")
    except Exception as e:
        print(f" Failed to read data from database! Error: {e}")
        return

    print(" [2/5] Selecting feature vectors and isolating target label...")
    # Drop structural metadata text variables; keep raw numerical sensor readings
    features = ['Air_Temperature_K', 'Process_Temperature_K', 'Rotational_Speed_RPM', 'Torque_Nm', 'Tool_Wear_Mins']
    X = df[features]
    y = df['Target']  # 0 = Normal running, 1 = Mechanical failure

    print(" [3/5] Partitioning data into Train/Test matrix split (80/20)...")
    # stratify=y is critical for imbalanced data to keep equal failure distribution in both sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print(" [4/5] Training Random Forest Classification Engine (Balancing Classes)...")
    # class_weight='balanced' automatically recalibrates weights for our imbalanced failure data
    model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)
    print(" Model training cycle complete.")

    print(" [5/5] Generating Predictive Operational Metrics...")
    y_pred = model.predict(X_test)
    y_probs = model.predict_proba(X_test)[:, 1]  # Failure probability scores

    # Output evaluation performance report to terminal
    print("\n======================= MODEL PERFORMANCE REPORT =======================")
    print(classification_report(y_test, y_pred))
    print("========================================================================\n")

    print(" Appending real-time failure risk scores back to warehouse schema...")
    # Assign the calculated anomaly metrics directly to the test evaluation cohort
    scored_test_df = X_test.copy()
    scored_test_df['CustomerID_Stub'] = df.loc[X_test.index, 'ProductID']  # Tracking mapping link
    scored_test_df['actual_failure_status'] = y_test
    scored_test_df['predicted_failure_status'] = y_pred
    scored_test_df['failure_probability_score'] = y_probs

    try:
        # Save results into a fresh analytical table inside your database for Power BI
        scored_test_df.to_sql(name="pred_machinery_risk_scores", con=engine, if_exists="replace", index=False)
        print(" SUCCESS! All machine anomaly scoring metrics saved to 'pred_machinery_risk_scores'.")
    except Exception as e:
        print(f" Failed to write scores to database! Error: {e}")

if __name__ == "__main__":
    train_failure_predictor()