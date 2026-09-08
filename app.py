import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge

# ----------------------------------------------------------------------
# Page Configuration & UI Theme Setup
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="EV Range Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Light Beige Theme with High-Contrast Typography
st.markdown("""
<style>
    /* Remove unnecessary padding at top */
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 2rem !important;
    }
    header[data-testid="stHeader"] {
        background: transparent !important;
        height: 0px !important;
    }
    
    /* Beige App Background */
    .stApp {
        background-color: #F5F2EB !important;
        color: #1E293B !important;
    }
    
    /* Global Text Colors for Dark Contrast */
    h1, h2, h3, h4, h5, h6, p, label, span {
        color: #0F172A !important;
    }
    
    /* Sidebar Styling - Soft Cream Beige */
    section[data-testid="stSidebar"] {
        background-color: #EBE5D8 !important;
        border-right: 1px solid #D6CEBE !important;
    }
    
    section[data-testid="stSidebar"] label p, 
    section[data-testid="stSidebar"] div, 
    section[data-testid="stSidebar"] span {
        color: #0F172A !important;
        font-weight: 600 !important;
    }
    
    /* Sidebar Headers */
    .sidebar-header {
        color: #0284C7 !important;
        font-size: 14px !important;
        font-weight: 700 !important;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 15px;
        margin-bottom: 5px;
    }

    /* Metric Values Typography */
    div[data-testid="stMetricValue"] {
        font-size: 42px !important;
        font-weight: 800 !important;
        color: #059669 !important;
    }
    
    /* Metric Card Labels */
    div[data-testid="stMetricLabel"],
    div[data-testid="stMetricLabel"] label,
    div[data-testid="stMetricLabel"] div,
    div[data-testid="stMetricLabel"] p,
    div[data-testid="stMetricLabel"] span {
        color: #334155 !important;
        font-size: 15px !important;
        font-weight: 700 !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
</style>
""", unsafe_allow_html=True)

DATA_PATH = "ev_data.xls"
RANDOM_STATE = 42

# ----------------------------------------------------------------------
# 1. Data Cleaning & Feature Engineering
# ----------------------------------------------------------------------
@st.cache_data
def load_and_clean_data(path: str) -> pd.DataFrame:
    raw_df = pd.read_excel(path)
    df = raw_df.copy()

    drop_columns = ["model", "source_url", "battery_type", "fast_charge_port", "number_of_cells"]
    drop_columns = [c for c in drop_columns if c in df.columns]
    df.drop(columns=drop_columns, inplace=True)

    if "cargo_volume_l" in df.columns:
        df["cargo_volume_l"] = pd.to_numeric(df["cargo_volume_l"], errors="coerce")

    if "towing_capacity_kg" in df.columns:
        df["towing_capacity_missing"] = df["towing_capacity_kg"].isnull().astype(int)

    df.drop_duplicates(inplace=True)

    if "efficiency_wh_per_km" in df.columns:
        df.drop(columns=["efficiency_wh_per_km"], inplace=True)

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["footprint_m2"] = (df["length_mm"] * df["width_mm"]) / 1e6
    df["vehicle_volume_m3"] = (df["length_mm"] * df["width_mm"] * df["height_mm"]) / 1e9
    df["battery_per_seat"] = df["battery_capacity_kWh"] / df["seats"]
    df["charge_power_ratio"] = df["fast_charging_power_kw_dc"] / df["battery_capacity_kWh"]
    df["torque_per_battery"] = df["torque_nm"] / df["battery_capacity_kWh"]
    df["speed_acceleration_ratio"] = df["top_speed_kmh"] / df["acceleration_0_100_s"]
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df


# ----------------------------------------------------------------------
# 2. Model Training
# ----------------------------------------------------------------------
@st.cache_resource
def train_model(path: str):
    df = load_and_clean_data(path)
    df_model = engineer_features(df)

    numeric_columns = df_model.select_dtypes(include=np.number).columns.tolist()
    numeric_columns.remove("range_km")
    for col in numeric_columns:
        q1, q99 = df_model[col].quantile(0.01), df_model[col].quantile(0.99)
        df_model[col] = df_model[col].clip(lower=q1, upper=q99)

    y = df_model["range_km"]
    X = df_model.drop(columns=["range_km"])

    numeric_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()

    preprocessor = ColumnTransformer([
        ("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), numeric_features),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical_features),
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("model", Ridge(alpha=10, random_state=RANDOM_STATE)),
    ])
    model.fit(X, y)

    return model, X, df


def predict_range(model, X_columns, raw_input: dict) -> float:
    data = pd.DataFrame([raw_input])
    data = engineer_features(data)
    data = data[X_columns]
    pred = model.predict(data)[0]
    return max(0.0, float(pred))


# Load model
try:
    model, X_columns_df, raw_df = train_model(DATA_PATH)
    X_columns = X_columns_df.columns
except FileNotFoundError:
    st.error(f"Couldn't find `{DATA_PATH}`. Please check if the file exists in the directory.")
    st.stop()


# ----------------------------------------------------------------------
# 3. Sidebar Controls
# ----------------------------------------------------------------------
st.sidebar.title("Vehicle Controls")

st.sidebar.markdown("<p class='sidebar-header'>Basic Specifications</p>", unsafe_allow_html=True)
brand = st.sidebar.selectbox("Brand / Manufacturer", sorted(raw_df["brand"].dropna().unique()))
drivetrain = st.sidebar.selectbox("Drivetrain", sorted(raw_df["drivetrain"].dropna().unique()))
segment = st.sidebar.selectbox("Segment", sorted(raw_df["segment"].dropna().unique()))
car_body_type = st.sidebar.selectbox("Body Type", sorted(raw_df["car_body_type"].dropna().unique()))

st.sidebar.markdown("<p class='sidebar-header'>Performance & Dimensions</p>", unsafe_allow_html=True)

def slider_from_data(label, col, step=1.0, fmt=None):
    lo = float(raw_df[col].quantile(0.01))
    hi = float(raw_df[col].quantile(0.99))
    default = float(raw_df[col].median())
    return st.sidebar.slider(label, lo, hi, default, step=step, format=fmt)

battery_capacity_kWh = slider_from_data("Battery Capacity (kWh)", "battery_capacity_kWh", step=0.5)
top_speed_kmh = slider_from_data("Top Speed (km/h)", "top_speed_kmh")
torque_nm = slider_from_data("Torque (Nm)", "torque_nm")
acceleration_0_100_s = slider_from_data("0–100 km/h (s)", "acceleration_0_100_s", step=0.1)
fast_charging_power_kw_dc = slider_from_data("Fast Charge Power (kW)", "fast_charging_power_kw_dc")
towing_capacity_kg = slider_from_data("Towing Capacity (kg)", "towing_capacity_kg")
cargo_volume_l = slider_from_data("Cargo Volume (L)", "cargo_volume_l")
seats = st.sidebar.slider("Seats", 2, 9, 5, step=1)
length_mm = slider_from_data("Length (mm)", "length_mm")
width_mm = slider_from_data("Width (mm)", "width_mm")
height_mm = slider_from_data("Height (mm)", "height_mm")


# ----------------------------------------------------------------------
# 4. Main Interface
# ----------------------------------------------------------------------
st.title("Electric Vehicle Range Estimator")
st.caption("Estimate real-world driving range based on vehicle specifications and battery size.")

# Construct Input Payload
raw_input = {
    "brand": brand,
    "top_speed_kmh": top_speed_kmh,
    "battery_capacity_kWh": battery_capacity_kWh,
    "torque_nm": torque_nm,
    "acceleration_0_100_s": acceleration_0_100_s,
    "fast_charging_power_kw_dc": fast_charging_power_kw_dc,
    "towing_capacity_kg": towing_capacity_kg,
    "towing_capacity_missing": 0,
    "cargo_volume_l": cargo_volume_l,
    "seats": seats,
    "drivetrain": drivetrain,
    "segment": segment,
    "length_mm": length_mm,
    "width_mm": width_mm,
    "height_mm": height_mm,
    "car_body_type": car_body_type,
}

prediction = predict_range(model, X_columns, raw_input)

st.markdown("<br>", unsafe_allow_html=True)

# Animated Driving Track - Adapted for Light Theme
max_benchmark_range = 700.0
progress_percent = min(100, max(0, int((prediction / max_benchmark_range) * 100)))

car_track_html = f"""
<div style="background: #EBE5D8; padding: 24px; border-radius: 16px; border: 1px solid #D6CEBE; margin-bottom: 25px; box-shadow: 0 4px 12px rgba(0,0,0,0.05);">
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;">
        <span style="color: #0284C7; font-size: 13px; font-weight: 700; text-transform: uppercase; letter-spacing: 1.5px;">Simulated Driving Test — {brand}</span>
        <span style="color: #059669; font-weight: 800; font-size: 16px; background: rgba(5, 150, 105, 0.1); padding: 4px 12px; border-radius: 20px; border: 1px solid rgba(5, 150, 105, 0.3);">
            Ready
        </span>
    </div>
    
    <div style="position: relative; height: 125px; border-bottom: 4px dashed #94A3B8; margin: 0 50px 15px 50px;">
        <div style="
            position: absolute; 
            left: {progress_percent}%; 
            transform: translateX(-50%); 
            bottom: 0px; 
            transition: left 2.5s cubic-bezier(0.4, 0, 0.2, 1);
            will-change: left;
            display: flex;
            flex-direction: column;
            align-items: center;
        ">
            <!-- Speech Bubble -->
            <div style="
                background: #059669; 
                color: #FFFFFF; 
                padding: 6px 14px; 
                border-radius: 20px; 
                font-weight: 800; 
                font-size: 13px; 
                white-space: nowrap;
                box-shadow: 0 4px 10px rgba(5, 150, 105, 0.3);
                margin-bottom: 8px;
                position: relative;
                font-family: sans-serif;
            ">
                Estimated Range: {prediction:.0f} km
                <div style="
                    position: absolute;
                    bottom: -6px;
                    left: 50%;
                    transform: translateX(-50%);
                    width: 0;
                    height: 0;
                    border-left: 6px solid transparent;
                    border-right: 6px solid transparent;
                    border-top: 6px solid #059669;
                "></div>
            </div>
            
            <!-- SVG Car Graphic -->
            <svg width="105" height="52" viewBox="0 0 100 50" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M85 35H95V24C95 22.2 93.5 20.8 91.8 20.8H80L69 12C67.3 10.7 65.2 10 63 10H30C26.7 10 24 12.7 24 16V20.8H12C8.7 20.8 6 23.5 6 26.8V35H18" stroke="#0284C7" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" fill="#F5F2EB"/>
                <circle cx="28" cy="36" r="6" fill="#0F172A" stroke="#F5F2EB" stroke-width="3"/>
                <circle cx="75" cy="36" r="6" fill="#0F172A" stroke="#F5F2EB" stroke-width="3"/>
                <path d="M28 20.8V15C28 14.4 28.4 14 29 14H61C61.8 14 62.5 14.4 63 15L70 20.8H28Z" fill="#0284C7" opacity="0.3"/>
                <text x="50" y="29" font-family="sans-serif" font-size="6.5" font-weight="900" fill="#059669" text-anchor="middle" letter-spacing="0.5">POWER PREDICTORS</text>
            </svg>
        </div>
    </div>
    
    <div style="display: flex; justify-content: space-between; margin: 0 30px; color: #64748B; font-size: 12px; font-weight: 700;">
        <span>0 km</span>
        <span>175 km</span>
        <span>350 km</span>
        <span>525 km</span>
        <span>700+ km</span>
    </div>
</div>
"""

st.components.v1.html(car_track_html, height=235)

# Metrics Display
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Predicted Range", f"{prediction:.0f} km")

with col2:
    est_efficiency = (battery_capacity_kWh / (prediction + 1e-5)) * 1000
    st.metric("Est. Efficiency", f"{est_efficiency:.0f} Wh/km")

with col3:
    st.metric("Battery Capacity", f"{battery_capacity_kWh:.1f} kWh")

st.markdown("<br>", unsafe_allow_html=True)

# ----------------------------------------------------------------------
# 5. Range vs Battery Curve Chart
# ----------------------------------------------------------------------
st.subheader("Range vs. Battery Capacity Curve")
st.caption("How predicted range scales as battery capacity increases while holding other parameters constant.")

sweep_vals = np.linspace(
    raw_df["battery_capacity_kWh"].quantile(0.01),
    raw_df["battery_capacity_kWh"].quantile(0.99),
    35,
)
sweep_preds = [
    predict_range(model, X_columns, {**raw_input, "battery_capacity_kWh": v})
    for v in sweep_vals
]

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=sweep_vals,
    y=sweep_preds,
    mode='lines',
    name='Predicted Range',
    line=dict(color='#059669', width=4, shape='spline'),
    fill='tozeroy',
    fillcolor='rgba(5, 150, 105, 0.1)'
))

fig.add_trace(go.Scatter(
    x=[battery_capacity_kWh],
    y=[prediction],
    mode='markers+text',
    name='Current Setup',
    marker=dict(color='#0284C7', size=14, line=dict(color='#FFFFFF', width=2)),
    text=[f"{prediction:.0f} km"],
    textposition="top center",
    textfont=dict(color="#0F172A", size=13, family="sans-serif")
))

fig.update_layout(
    paper_bgcolor='rgba(245, 242, 235, 0)',
    plot_bgcolor='rgba(235, 229, 216, 0.5)',
    xaxis=dict(
        title=dict(text="Battery Capacity (kWh)", font=dict(color='#334155')),
        gridcolor='#D6CEBE',
        zerolinecolor='#D6CEBE',
        tickfont=dict(color='#334155')
    ),
    yaxis=dict(
        title=dict(text="Predicted Range (km)", font=dict(color='#334155')),
        gridcolor='#D6CEBE',
        zerolinecolor='#D6CEBE',
        tickfont=dict(color='#334155')
    ),
    margin=dict(l=20, r=20, t=20, b=20),
    showlegend=False,
    height=340
)

st.plotly_chart(fig, use_container_width=True)