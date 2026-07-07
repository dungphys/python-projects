import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import numpy as np
import os

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="PhysicsNet", page_icon="⚛️", layout="wide")
st.title("⚛️ PhysicsNet — Particle Classifier Dashboard")

# ── Sidebar: Manual Input ────────────────────────────────────
st.sidebar.header("Particle Features")
energy     = st.sidebar.slider("Energy (GeV)",     1.0,  500.0, 85.0)
momentum_x = st.sidebar.slider("Momentum X",     -200.0, 200.0,  12.0)
momentum_y = st.sidebar.slider("Momentum Y",     -200.0, 200.0,  -5.0)
momentum_z = st.sidebar.slider("Momentum Z",     -200.0, 200.0,  30.0)
charge     = st.sidebar.selectbox("Charge",      [-1.0, 0.0, 1.0])
hit_0      = st.sidebar.number_input("Hit Layer 0", 0, 50, 8)
hit_1      = st.sidebar.number_input("Hit Layer 1", 0, 50, 9)
hit_2      = st.sidebar.number_input("Hit Layer 2", 0, 50, 7)
hit_3      = st.sidebar.number_input("Hit Layer 3", 0, 50, 10)

payload = dict(
    energy=energy, momentum_x=momentum_x, momentum_y=momentum_y,
    momentum_z=momentum_z, charge=charge,
    hit_0=hit_0, hit_1=hit_1, hit_2=hit_2, hit_3=hit_3,
)

# ── Main Panel ───────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("Single Prediction")
    if st.button("🔍 Classify Particle", use_container_width=True):
        with st.spinner("Running inference..."):
            resp = requests.post(f"{API_URL}/predict", json=payload)
            if resp.status_code == 200:
                result = resp.json()
                st.success(f"**Predicted:** {result['particle'].upper()}")
                st.metric("Confidence", f"{result['confidence']:.1%}")
                proba_df = pd.DataFrame.from_dict(
                    result["probabilities"], orient="index", columns=["probability"]
                ).reset_index().rename(columns={"index": "particle"})
                fig = px.bar(proba_df, x="particle", y="probability",
                             color="particle", title="Class Probabilities")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error(f"API error: {resp.text}")

with col2:
    st.subheader("Batch Simulation")
    n_batch = st.slider("Number of random events", 10, 200, 50)
    if st.button("🎲 Run Batch Simulation", use_container_width=True):
        rng = np.random.default_rng()
        batch = [dict(
            energy=float(rng.exponential(100)),
            momentum_x=float(rng.normal(0, 30)),
            momentum_y=float(rng.normal(0, 30)),
            momentum_z=float(rng.normal(0, 30)),
            charge=float(rng.choice([-1, 0, 1])),
            hit_0=int(rng.poisson(9)), hit_1=int(rng.poisson(9)),
            hit_2=int(rng.poisson(9)), hit_3=int(rng.poisson(9)),
        ) for _ in range(n_batch)]

        with st.spinner(f"Classifying {n_batch} events..."):
            resp = requests.post(f"{API_URL}/predict/batch", json=batch)
            if resp.status_code == 200:
                results   = resp.json()
                particles = [r["particle"] for r in results]
                counts_df = pd.Series(particles).value_counts().reset_index()
                counts_df.columns = ["particle", "count"]
                fig = px.pie(counts_df, names="particle", values="count",
                             title="Particle Distribution")
                st.plotly_chart(fig, use_container_width=True)

# ── API Health ───────────────────────────────────────────────
st.divider()
try:
    health = requests.get(f"{API_URL}/health", timeout=2).json()
    st.success(f"API Status: {health['status']}")
except Exception:
    st.error("API not reachable")