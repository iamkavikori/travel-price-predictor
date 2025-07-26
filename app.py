import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor

# 🔹 Generate synthetic dataset (small & fast to train)
@st.cache_data
def load_data():
    np.random.seed(42)
    airlines = ['IndiGo', 'Air India', 'SpiceJet', 'Vistara', 'GoAir']
    sources = ['Delhi', 'Mumbai', 'Chennai', 'Bangalore', 'Kolkata']
    destinations = ['Delhi', 'Mumbai', 'Chennai', 'Bangalore', 'Kolkata']
    
    data = {
        'Airline': np.random.choice(airlines, 300),
        'Source': np.random.choice(sources, 300),
        'Destination': np.random.choice(destinations, 300),
        'Duration': np.random.randint(60, 1200, 300),
        'Stops': np.random.randint(0, 3, 300),
        'Price': np.random.randint(2500, 25000, 300)
    }
    return pd.DataFrame(data)

# 🔹 Train model dynamically
@st.cache_resource
def train_model(df):
    df = df.copy()
    categorical_cols = ['Airline', 'Source', 'Destination']
    encoders = {}

    for col in categorical_cols:
        enc = LabelEncoder()
        df[col] = enc.fit_transform(df[col])
        encoders[col] = enc

    X = df[['Airline', 'Source', 'Destination', 'Duration', 'Stops']]
    y = df['Price']

    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X, y)
    return model, encoders

# 🔹 Load data & train
df = load_data()
model, encoders = train_model(df)

# 🔹 Dropdown Options
airlines = df['Airline'].unique().tolist()
sources = df['Source'].unique().tolist()
destinations = df['Destination'].unique().tolist()

# 🔹 Streamlit UI
st.set_page_config(page_title="Travel Price Predictor", page_icon="✈️", layout="centered")
st.title("✈️ Travel Price Predictor")
st.markdown("An AI/ML web app to estimate flight prices based on your trip details.")

col1, col2 = st.columns(2)

with col1:
    airline = st.selectbox("Select Airline", airlines)
    source = st.selectbox("Select Source", sources)

with col2:
    destination = st.selectbox("Select Destination", destinations)
    duration = st.slider("Duration (minutes)", 30, 1500, 180)

stops = st.slider("Total Stops", 0, 2, 1)

# 🔹 Prediction
if st.button("Predict Price"):
    a = encoders['Airline'].transform([airline])[0]
    s = encoders['Source'].transform([source])[0]
    d = encoders['Destination'].transform([destination])[0]

    sample = np.array([[a, s, d, duration, stops]])
    price = model.predict(sample)[0]
    st.success(f"💰 Estimated Flight Price: ₹ {round(price,2)}")

st.markdown("---")
st.caption("🚀 Built by Kavita Kori | Powered by Streamlit & RandomForestRegressor")
