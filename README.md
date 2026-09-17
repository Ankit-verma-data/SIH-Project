# SIH-Project
Forecast of Expected Time of Arrival (ETA) for Coaching Trains

# 🚆 Dynamic Train ETA Predictor & Live Tracking System
### Smart India Hackathon (SIH 2026) | Ministry of Railways

A real-time machine learning-powered web application built to predict dynamic train delays, calculate accurate Expected Time of Arrival (ETA), and simulate live GPS tracking for optimized railway operations.

---

## 🚀 Key Features
- **Real-Time ML Predictions:** Uses an optimized `GradientBoostingRegressor` model trained on historical railway data to compute expected delay minutes.
- **Simulated Live GPS Tracking:** Connects with mock live coordinates and speed tracking to adjust ETA dynamically.
- **Interactive Scenarios:** Built-in quick demo presets for emergency situations like *Foggy Mornings* and *Peak Hour Rush*.
- **Domain Guardrails:** Integrated heuristic adjustments for train types (Superfast, Express, Local) and route congestion levels.
- **Interactive Visualizations:** Live mapping using `st.map` and dynamic analytical charts.

---

## 🛠️ Tech Stack
- **Frontend & Dashboard:** Streamlit
- **Machine Learning:** Scikit-Learn (Gradient Boosting Regressor)
- **Data Manipulation:** Pandas, NumPy

---

## 📂 Project Structure
```text
├── app.py               # Main Streamlit application
├── requirements.txt     # Required python packages
└── README.md            # Project documentation
