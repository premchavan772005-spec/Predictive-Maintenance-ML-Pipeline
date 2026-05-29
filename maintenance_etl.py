import pandas as pd
import pymysql
from sqlalchemy import create_engine

# --- CONFIGURATION SETTINGS ---
# ⚠️ Update 'your_password_here' with your actual local MySQL root password!
DB_USER = "root"
DB_PASSWORD = "its_prem7725$67" 
DB_HOST = "localhost"
DB_PORT = "3306"
DB_NAME = "predictive_maintenance_db"
TABLE_NAME = "fact_sensor_logs"

# Clean relative path since the file sits right next to this script!
CSV_PATH = "predictive_maintenance.csv"

def run_sensor_etl():
    print("⏳ [1/4] Extracting raw telemetry dataset from local project folder...")
    try:
        df = pd.read_csv(CSV_PATH)
        print(f"✅ Extraction complete. Found {df.shape[0]} data log streams.")
    except Exception as e:
        print(f"❌ Extraction failed! Check your file path. Error: {e}")
        return

    print("🧹 [2/4] Initializing data cleaning & mapping transformations...")
    # Standardizing messy raw column names to match our precise MySQL warehouse schema
    column_mapping = {
        'UDI': 'UDI',
        'Product ID': 'ProductID',
        'Type': 'Type',
        'Air temperature [K]': 'Air_Temperature_K',
        'Process temperature [K]': 'Process_Temperature_K',
        'Rotational speed [rpm]': 'Rotational_Speed_RPM',
        'Torque [Nm]': 'Torque_Nm',
        'Tool wear [min]': 'Tool_Wear_Mins',
        'Target': 'Target',
        'Failure Type': 'Failure_Type'
    }
    df.rename(columns=column_mapping, inplace=True)
    
    # Keep only the columns that match our SQL table structure
    df = df[list(column_mapping.values())]
    print("✅ Schema alignment and renaming operations successful.")

    print("🔌 [3/4] Establishing active SQLAlchemy database connection...")
    try:
        connection_string = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
        engine = create_engine(connection_string)
        print("✅ Database connection validated.")
    except Exception as e:
        print(f"❌ Database connection failed! Check credentials. Error: {e}")
        return

    print(f"🚀 [4/4] Streaming sensor records directly into target '{TABLE_NAME}' warehouse...")
    try:
        # if_exists='append' ensures it inputs data directly into the table we built
        df.to_sql(name=TABLE_NAME, con=engine, if_exists='append', index=False)
        print(f"🏆 SUCCESS! All {df.shape[0]} sensor records safely integrated into your warehouse table.")
    except Exception as e:
        print(f"❌ Loading data failed! Error: {e}")

if __name__ == "__main__":
    run_sensor_etl()