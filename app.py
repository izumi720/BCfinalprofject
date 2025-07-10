# app.py
import streamlit as st
import pandas as pd
import joblib
import sqlite3
import os

# --- Database Setup ---
DATABASE_FILE = 'diamonds.db'

def init_db():
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS diamonds (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            carat REAL,
            cut TEXT,
            color TEXT,
            clarity TEXT,
            depth REAL,
            "table" REAL,
            price REAL,
            x REAL,
            y REAL,
            z REAL
        )
    ''')
    conn.commit()
    conn.close()

def insert_diamond_data(data):
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO diamonds (carat, cut, color, clarity, depth, "table", price, x, y, z)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (data['carat'], data['cut'], data['color'], data['clarity'], data['depth'],
          data['table'], data['price'], data['x'], data['y'], data['z']))
    conn.commit()
    conn.close()

def get_all_diamond_data():
    conn = sqlite3.connect(DATABASE_FILE)
    df = pd.read_sql_query("SELECT * FROM diamonds", conn)
    conn.close()
    return df

# Initialize the database when the app starts
if not os.path.exists(DATABASE_FILE):
    init_db()
    # Optionally, load initial data from diamonds2.csv into the database
    # This part assumes you have diamonds2.csv in the same directory
    try:
        initial_df = pd.read_csv('diamonds2.csv')
        # Clean up column names if necessary (e.g., remove leading '""')
        initial_df.columns = [col.replace('"', '') for col in initial_df.columns]
        # Drop the first unnamed column if it exists
        if '' in initial_df.columns:
            initial_df = initial_df.drop(columns=[''])

        conn = sqlite3.connect(DATABASE_FILE)
        initial_df.to_sql('diamonds', conn, if_exists='append', index=False)
        conn.close()
        st.success("Initial data loaded into database.")
    except FileNotFoundError:
        st.warning("diamonds2.csv not found. Database initialized empty.")
    except Exception as e:
        st.error(f"Error loading initial data: {e}")


# --- Streamlit App ---
st.title("💎 Diamond Price Predictor")

# Load model and feature list (these are still loaded from files)
try:
    model = joblib.load('model.pkl')
    features = joblib.load('features.pkl')
except FileNotFoundError:
    st.error("Model files (model.pkl or features.pkl) not found. Please ensure they are in the same directory.")
    st.stop() # Stop the app if model files are missing

# Input fields for diamond features
carat = st.number_input("Carat", 0.0, 5.0, 0.5)
depth = st.number_input("Depth", 50.0, 70.0, 61.5)
table = st.number_input("Table", 50.0, 70.0, 55.0)
x = st.number_input("X", 0.0, 10.0, 5.1)
y = st.number_input("Y", 0.0, 10.0, 5.1)
z = st.number_input("Z", 0.0, 10.0, 3.1)

cut = st.selectbox("Cut", ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal'])
color = st.selectbox("Color", ['D', 'E', 'F', 'G', 'H', 'I', 'J'])
clarity = st.selectbox("Clarity", ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF'])

if st.button("Predict"):
    input_data = {
        'carat': carat,
        'depth': depth,
        'table': table,
        'x': x,
        'y': y,
        'z': z,
        'cut': cut,
        'color': color,
        'clarity': clarity
    }
    input_df = pd.DataFrame([input_data])
    input_encoded = pd.get_dummies(input_df)
    input_encoded = input_encoded.reindex(columns=features, fill_value=0)
    prediction = model.predict(input_encoded)[0]
    st.success(f"Estimated Price: ${prediction:,.2f}")

    # Save the prediction and input data to the database
    input_data['price'] = prediction
    insert_diamond_data(input_data)
    st.info("Prediction saved to database.")

st.subheader("Recent Diamond Predictions")
db_data = get_all_diamond_data()
if not db_data.empty:
    st.dataframe(db_data.tail(10)) # Display last 10 entries
else:
    st.info("No data in the database yet.")

