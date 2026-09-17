#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split


# In[16]:


df= pd.read_csv("train_delay_dataset.zip")


# In[3]:


df.head()


# In[4]:


df["Weather Conditions"].nunique()
df["Day of the Week"].nunique()
df["Time of Day"].nunique()
df["Train Type"].nunique()
df["Route Congestion"].nunique()
df["Historical Delay (min)"].nunique()


# In[5]:


x= df.drop(columns=["Historical Delay (min)"])
y=df["Historical Delay (min)"]


# In[6]:


# one hot encoding
x= pd.get_dummies(x, columns=["Weather Conditions", "Day of the Week", "Time of Day", "Train Type", "Route Congestion"], drop_first= True, dtype=int)
x


# In[7]:


x_train, x_test, y_train, y_test= train_test_split(x, y, test_size=0.2, random_state=42)


# In[8]:


from sklearn.ensemble import GradientBoostingRegressor
print(x.dtypes)


# In[9]:


gbr= GradientBoostingRegressor()
gbr.fit(x_train, y_train)
y_pred= gbr.predict(x_test)


# In[13]:


# Evaluation
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
r2=r2_score(y_test, y_pred)
print("R2_score:", r2)

mae=mean_absolute_error(y_test, y_pred)
print("MAE:", mae)

MSE=mean_squared_error(y_test, y_pred)
print("MSE:", MSE)


# ### Streamlit UI

# In[19]:


# ============================================================
# 🚆 DYNAMIC TRAIN ETA PREDICTOR
# Smart India Hackathon 2026
# ============================================================

# -------------------- 1. IMPORT LIBRARIES --------------------

import time
import random
import pandas as pd
import numpy as np
import streamlit as st

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score


# -------------------- 2. PAGE CONFIGURATION --------------------

st.set_page_config(
    page_title="Dynamic Train ETA Predictor - SIH 2026",
    page_icon="🚆",
    layout="wide"
)


# -------------------- 3. TRAIN ML MODEL --------------------

@st.cache_resource
def load_and_train_model():

    # Load dataset with fallback safety if zip file is missing
    try:
        df = pd.read_csv("train_delay_dataset.zip")
    except FileNotFoundError:
        np.random.seed(42)
        n = 1000
        df = pd.DataFrame({
            "Distance Between Stations (km)": np.random.randint(10, 1000, n),
            "Weather Conditions": np.random.choice(["Clear", "Foggy", "Rainy"], n),
            "Day of the Week": np.random.choice(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"], n),
            "Time of Day": np.random.choice(["Morning", "Afternoon", "Evening", "Night"], n),
            "Train Type": np.random.choice(["Express", "Superfast", "Local"], n),
            "Route Congestion": np.random.choice(["Low", "Medium", "High"], n),
            "Historical Delay (min)": np.random.uniform(0, 50, n)
        })
        st.warning("⚠️ 'train_delay_dataset.zip' not found. Automatically loaded synthetic sample data for smooth execution.")

    # Target variable
    target = "Historical Delay (min)"

    # Separate input and output
    X = df.drop(columns=[target])
    y = df[target]

    # Convert categorical data into numerical data
    categorical_columns = [
        "Weather Conditions",
        "Day of the Week",
        "Time of Day",
        "Train Type",
        "Route Congestion"
    ]

    X_encoded = pd.get_dummies(
        X,
        columns=categorical_columns,
        drop_first=False
    )

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded,
        y,
        test_size=0.20,
        random_state=42
    )

    # Create model
    model = GradientBoostingRegressor(
        random_state=42
    )

    # Train model
    model.fit(X_train, y_train)

    # Test model
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    return model, X_encoded.columns, mae, r2


# Load trained model
model, model_columns, mae, r2 = load_and_train_model()


# -------------------- 4. MOCK LIVE GPS FUNCTION --------------------

def fetch_mock_live_gps():

    # Delhi coordinates used for demonstration
    latitude = 28.6139 + random.uniform(-0.02, 0.02)
    longitude = 77.2090 + random.uniform(-0.02, 0.02)

    # Simulated train speed
    live_speed = random.randint(40, 90)

    return latitude, longitude, live_speed


# -------------------- 5. HEADER --------------------

st.title("🚆 Dynamic Forecast of Expected Time of Arrival (ETA)")

st.markdown(
    "**Smart India Hackathon 2026 | Ministry of Railways**  \n"
    "Real-Time Train Delay & ETA Prediction System"
)

st.markdown("---")


# ============================================================
# 6. SESSION STATE
# ============================================================

if "preset_weather" not in st.session_state:
    st.session_state.preset_weather = "Clear"

if "preset_congestion" not in st.session_state:
    st.session_state.preset_congestion = "Low"

if "preset_train_type" not in st.session_state:
    st.session_state.preset_train_type = "Express"

if "preset_distance" not in st.session_state:
    st.session_state.preset_distance = 100

if "preset_day" not in st.session_state:
    st.session_state.preset_day = "Monday"

if "preset_time" not in st.session_state:
    st.session_state.preset_time = "Morning"

if "preset_station" not in st.session_state:
    st.session_state.preset_station = "Origin (Stop 0)"


# ============================================================
# 7. SIDEBAR
# ============================================================

st.sidebar.header("⚡ Quick Demo Scenarios")

st.sidebar.write(
    "Use these buttons to quickly demonstrate different situations."
)


# -------------------- Scenario 1 --------------------

col_s1, col_s2 = st.sidebar.columns(2)

if col_s1.button("🌫️ Foggy Morning"):

    st.session_state.preset_weather = "Foggy"
    st.session_state.preset_congestion = "High"
    st.session_state.preset_train_type = "Express"
    st.session_state.preset_distance = 350
    st.session_state.preset_day = "Monday"
    st.session_state.preset_time = "Morning"
    st.session_state.preset_station = "Origin (Stop 0)"


# -------------------- Scenario 2 --------------------

if col_s2.button("🔥 Peak Hour Rush"):

    st.session_state.preset_weather = "Clear"
    st.session_state.preset_congestion = "High"
    st.session_state.preset_train_type = "Express"
    st.session_state.preset_distance = 150
    st.session_state.preset_day = "Monday"
    st.session_state.preset_time = "Evening"
    st.session_state.preset_station = "Intermediate (Stop 1)"


# -------------------- Scenario 3 --------------------

if st.sidebar.button("🟢 Normal Clear Run"):

    st.session_state.preset_weather = "Clear"
    st.session_state.preset_congestion = "Low"
    st.session_state.preset_train_type = "Superfast"
    st.session_state.preset_distance = 100
    st.session_state.preset_day = "Monday"
    st.session_state.preset_time = "Morning"
    st.session_state.preset_station = "Origin (Stop 0)"


st.sidebar.markdown("---")


# ============================================================
# 8. LIVE TRACKING
# ============================================================

st.sidebar.header("📡 Live Tracking Mode")

live_mode = st.sidebar.checkbox(
    "Enable Live GPS Simulation",
    value=True
)


st.sidebar.markdown("---")


# ============================================================
# 9. MODEL INFORMATION
# ============================================================

st.sidebar.header("🤖 ML Model Information")

st.sidebar.write("Model: Gradient Boosting Regressor")
st.sidebar.write(f"MAE: {mae:.2f} minutes")
st.sidebar.write(f"R² Score: {r2:.2f}")


# ============================================================
# 10. MAIN INPUT SECTION
# ============================================================

st.subheader("🎛️ Train Parameters")

col1, col2 = st.columns(2)


# -------------------- LEFT COLUMN --------------------

with col1:

    distance = st.slider(
        "📏 Distance Between Stations (km)",
        min_value=10,
        max_value=1000,
        value=st.session_state.preset_distance,
        step=10
    )

    weather_options = ["Clear", "Foggy", "Rainy"]

    weather = st.selectbox(
        "🌦️ Weather Conditions",
        weather_options,
        index=weather_options.index(st.session_state.preset_weather)
    )

    day_options = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    day_of_week = st.selectbox(
        "📅 Day of the Week",
        day_options,
        index=day_options.index(st.session_state.preset_day)
    )


# -------------------- RIGHT COLUMN --------------------

with col2:

    time_options = ["Morning", "Afternoon", "Evening", "Night"]

    time_of_day = st.selectbox(
        "🕐 Time of Day",
        time_options,
        index=time_options.index(st.session_state.preset_time)
    )

    train_options = ["Express", "Superfast", "Local"]

    train_type = st.selectbox(
        "🚆 Train Type",
        train_options,
        index=train_options.index(st.session_state.preset_train_type)
    )

    congestion_options = ["Low", "Medium", "High"]

    route_congestion = st.selectbox(
        "🚦 Route Congestion",
        congestion_options,
        index=congestion_options.index(st.session_state.preset_congestion)
    )


# ============================================================
# 11. STATION PROGRESS
# ============================================================

st.markdown("---")

station_options = [
    "Origin (Stop 0)",
    "Intermediate (Stop 1)",
    "Approaching Destination (Stop 2)"
]

station_step = st.select_slider(
    "📍 Route Progress Tracker",
    options=station_options,
    value=st.session_state.preset_station
)


# ============================================================
# 12. PREDICTION BUTTON
# ============================================================

st.markdown("---")

predict_button = st.button(
    "🚀 Predict Dynamic ETA & Delay",
    type="primary",
    use_container_width=True
)


# ============================================================
# 13. PREDICTION LOGIC
# ============================================================

if predict_button:

    latitude, longitude = 28.6139, 77.2090
    live_speed = 60

    if live_mode:

        with st.spinner("📡 Connecting to simulated GPS feed..."):
            time.sleep(0.8)
            latitude, longitude, live_speed = fetch_mock_live_gps()

        st.info(
            f"📍 **Live GPS Connected**  \n"
            f"Latitude: `{latitude:.4f}` | Longitude: `{longitude:.4f}` | Current Speed: `{live_speed} km/h`"
        )


    # --------------------------------------------------------
    # CREATE INPUT DATA
    # --------------------------------------------------------

    with st.spinner("🤖 Analyzing train conditions..."):

        time.sleep(0.4)

        input_data = pd.DataFrame({
            "Distance Between Stations (km)": [distance],
            "Weather Conditions": [weather],
            "Day of the Week": [day_of_week],
            "Time of Day": [time_of_day],
            "Train Type": [train_type],
            "Route Congestion": [route_congestion]
        })

        # ONE-HOT ENCODING
        input_encoded = pd.get_dummies(
            input_data,
            columns=[
                "Weather Conditions",
                "Day of the Week",
                "Time of Day",
                "Train Type",
                "Route Congestion"
            ],
            drop_first=False
        )

        input_encoded = input_encoded.reindex(
            columns=model_columns,
            fill_value=0
        )

        # ML PREDICTION
        predicted_delay = model.predict(input_encoded)[0]

        # ----------------------------------------------------
        # 14. DOMAIN GUARDRAILS
        # ----------------------------------------------------

        if train_type == "Superfast":
            predicted_delay *= 0.35
        elif train_type == "Express":
            predicted_delay *= 0.65

        if distance <= 30 and weather == "Clear" and route_congestion == "Low":
            predicted_delay = min(predicted_delay, 4.0)
        elif distance <= 50 and route_congestion == "Low":
            predicted_delay = min(predicted_delay, 7.0)

        if live_mode and live_speed >= 60 and route_congestion == "Low":
            predicted_delay = min(predicted_delay, 5.0)

        if live_mode and live_speed < 50:
            predicted_delay += (50 - live_speed) * 0.4

        if station_step == "Intermediate (Stop 1)":
            predicted_delay *= 1.10
        elif station_step == "Approaching Destination (Stop 2)":
            predicted_delay *= 1.20

        predicted_delay = max(0, round(predicted_delay, 1))

        # ----------------------------------------------------
        # 15. ETA CALCULATION
        # ----------------------------------------------------

        effective_speed = live_speed if live_mode else 60
        effective_speed = max(effective_speed, 20)

        travel_time = (distance / effective_speed) * 60
        estimated_eta_minutes = round(travel_time + predicted_delay, 1)


    # ========================================================
    # 16. DISPLAY RESULTS
    # ========================================================

    st.markdown("---")
    st.subheader("📊 Prediction Results")

    result1, result2, result3, result4 = st.columns(4)

    result1.metric("Predicted Delay", f"{predicted_delay} min")
    result2.metric("Travel Time", f"{travel_time:.1f} min")
    result3.metric("Dynamic ETA", f"{estimated_eta_minutes:.1f} min")
    result4.metric("Current Speed", f"{effective_speed} km/h")


    # ========================================================
    # 16.1 LIVE MAP VISUALIZATION (NEW FEATURE)
    # ========================================================

    if live_mode:
        st.markdown("---")
        st.subheader("🗺️ Live GPS Tracking Map")
        map_df = pd.DataFrame({"lat": [latitude], "lon": [longitude]})
        st.map(map_df, zoom=10, use_container_width=True)


    # ========================================================
    # 17. TRAIN STATUS
    # ========================================================

    st.markdown("---")

    if predicted_delay > 30:
        st.error(f"🚨 **HIGH DELAY WARNING**\n\nExpected delay: **{predicted_delay} minutes**\n\nSevere route conditions may affect train operations.")
    elif predicted_delay > 15:
        st.warning(f"⚠️ **MODERATE DELAY ALERT**\n\nExpected delay: **{predicted_delay} minutes**")
    else:
        st.success(f"✅ **TRAIN RUNNING NEAR SCHEDULE**\n\nPredicted delay: **{predicted_delay} minutes**")


    # ========================================================
    # 18. CONDITION SUMMARY
    # ========================================================

    st.markdown("---")
    st.subheader("🔎 Current Train Conditions")

    c1, c2, c3, c4 = st.columns(4)
    c1.write(f"**🚆 Train Type:** {train_type}")
    c2.write(f"**🌦️ Weather:** {weather}")
    c3.write(f"**🚦 Congestion:** {route_congestion}")
    c4.write(f"**📍 Progress:** {station_step}")


    # ========================================================
    # 19. SIMPLE VISUALIZATION
    # ========================================================

    st.markdown("---")
    st.subheader("📈 Delay Analysis")

    chart_data = pd.DataFrame({
        "Parameter": ["Predicted Delay", "Travel Time", "Dynamic ETA"],
        "Minutes": [predicted_delay, travel_time, estimated_eta_minutes]
    })

    st.bar_chart(chart_data.set_index("Parameter"))


# ============================================================
# END OF PROGRAM
# ============================================================

