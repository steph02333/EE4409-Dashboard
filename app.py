
import os
import re
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

st.set_page_config(
    page_title="Plantation Monitoring Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# CUSTOM THEME / UI
# =========================================================
st.markdown("""
<style>
    .stApp {
        background-color: #f6f7f2;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #eef3e7 0%, #e4ecda 100%);
        border-right: 1px solid #d7dfcc;
    }

    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #2f4f2f;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        font-size: 1rem;
        color: #5f6f52;
        margin-bottom: 1.0rem;
    }

    .section-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #355e3b;
        margin-bottom: 0.6rem;
        margin-top: 0.2rem;
    }

    .insight-box {
        background: linear-gradient(135deg, #f1f6ed 0%, #edf4e7 100%);
        border-left: 6px solid #6b8e23;
        padding: 1rem 1.2rem;
        border-radius: 14px;
        color: #334d2f;
        font-size: 0.98rem;
        margin-top: 0.3rem;
        margin-bottom: 0.5rem;
    }

    .small-note {
        color: #70846b;
        font-size: 0.88rem;
    }

    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #dde6d2;
        padding: 10px 14px;
        border-radius: 16px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.03);
    }

    div[data-testid="stMetricLabel"] {
        color: #6b7d60;
        font-weight: 600;
    }

    div[data-testid="stMetricValue"] {
        color: #2f4f2f;
    }

    .block-container {
        padding-top: 1.4rem;
        padding-bottom: 1.4rem;
    }

    .sprout-card {
        background: linear-gradient(135deg, #f5fbef 0%, #eef7e4 100%);
        border: 1px solid #d6e3c7;
        border-radius: 18px;
        padding: 16px 18px;
        margin-bottom: 14px;
    }

    .sprout-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #2f4f2f;
        margin-bottom: 0.25rem;
    }

    .sprout-subtitle {
        color: #60725a;
        font-size: 0.92rem;
        margin-bottom: 0.4rem;
    }

    .chat-bubble-user {
        background: #dff0d2;
        border-radius: 14px 14px 4px 14px;
        padding: 10px 12px;
        margin: 8px 0 8px 40px;
        color: #294128;
    }

    .chat-bubble-bot {
        background: white;
        border: 1px solid #d9e6cf;
        border-radius: 14px 14px 14px 4px;
        padding: 10px 12px;
        margin: 8px 40px 8px 0;
        color: #314931;
    }

    .faq-chip {
        display: inline-block;
        padding: 6px 10px;
        border-radius: 999px;
        background: #e7f2dd;
        color: #355e3b;
        font-size: 0.85rem;
        margin-right: 6px;
        margin-bottom: 8px;
        border: 1px solid #d2e2c2;
    }
</style>
""", unsafe_allow_html=True)

# =========================================================
# DATA
# =========================================================
@st.cache_data
def load_data():
    local_path = r"D:\Users\TC FANG\Desktop\Dashboard_4409\plantation_soil_data (2).xlsm"
    repo_path = "plantation_soil_data (2).xlsm"
    file_path = local_path if os.path.exists(local_path) else repo_path

    df = pd.read_excel(file_path, engine="openpyxl")
    df["timestamp"] = pd.to_datetime(df["timestamp"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).copy()
    df["date"] = df["timestamp"].dt.date
    df["hour"] = df["timestamp"].dt.hour
    return df.sort_values(["plot_id", "timestamp"]).reset_index(drop=True)

df = load_data()

# =========================================================
# ML MODEL
# =========================================================
@st.cache_resource
def train_moisture_model(dataframe):
    model_df = dataframe.copy().sort_values(["plot_id", "timestamp"])
    model_df["next_hour_moisture"] = model_df.groupby("plot_id")["soil_moisture_pct"].shift(-1)
    model_df = model_df.dropna(subset=["next_hour_moisture"]).copy()

    le = LabelEncoder()
    model_df["plot_code"] = le.fit_transform(model_df["plot_id"])

    features = model_df[[
        "soil_moisture_pct", "soil_temp_c", "soil_ec_ds_m",
        "soil_ph", "rainfall_mm", "irrigation_mm", "hour", "plot_code"
    ]]
    target = model_df["next_hour_moisture"]

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        max_depth=8,
        min_samples_split=4
    )
    model.fit(features, target)
    return model, le

model, plot_encoder = train_moisture_model(df)

def predict_next_moisture(row):
    plot_code = int(plot_encoder.transform([row["plot_id"]])[0])
    x = pd.DataFrame([{
        "soil_moisture_pct": row["soil_moisture_pct"],
        "soil_temp_c": row["soil_temp_c"],
        "soil_ec_ds_m": row["soil_ec_ds_m"],
        "soil_ph": row["soil_ph"],
        "rainfall_mm": row["rainfall_mm"],
        "irrigation_mm": row["irrigation_mm"],
        "hour": row["timestamp"].hour,
        "plot_code": plot_code
    }])
    return float(model.predict(x)[0])

# =========================================================
# CHATBOT KNOWLEDGE
# =========================================================
FIGURE_EXPLANATIONS = {
    "fig1": "Figure 1 shows daily average soil moisture by plot. Moisture stays broadly within the mid-20% range, so the plots have similar average conditions. However, the line patterns still differ slightly from day to day, which suggests that water retention is not identical across plots.",
    "fig2": "Figure 2 is the soil moisture boxplot. The medians are similar, but the spread is wide, meaning short-term moisture conditions fluctuate a lot even when the averages look similar. This is useful because it shows why average values alone do not tell the full story.",
    "fig3": "Figure 3 shows daily total rainfall by plot. Rainfall is uneven across days and across plots, with Plot3 receiving the most rainfall overall. This confirms that water input is intermittent and spatially uneven.",
    "fig4": "Figure 4 overlays daily rainfall and daily average soil moisture for one plot. The main takeaway is that higher rainfall does not always produce a matching rise in soil moisture. This suggests that drainage, delayed absorption, and soil retention affect the final moisture response.",
    "fig5": "Figure 5 shows daily average soil EC. EC stays within a relatively narrow range, so salinity is fairly stable. When rainfall is higher, EC tends to dip slightly, which supports the idea of dilution by water input."
}

FAQ = {
    "what does this dashboard do": "This dashboard summarizes soil moisture, rainfall, EC, temperature, and pH across the plantation plots. It helps compare plots, track daily trends, and support irrigation-related interpretation.",
    "why is irrigation zero": "Irrigation is zero throughout the dataset, so this period is entirely rainfall-driven. That means the dashboard is showing natural water-input behaviour rather than controlled irrigation behaviour.",
    "what does the prediction mean": "The prediction card estimates the next-hour soil moisture using the current moisture, temperature, EC, pH, rainfall, irrigation, plot ID, and hour. It should be treated as a prototype forecasting tool, not a fully validated control model.",
    "what are the anomalies": "The main anomalies are very low soil moisture readings, occasional high EC spikes, uneven rainfall, and the weak rainfall-to-moisture relationship. These suggest that soil moisture is influenced by more than rainfall alone.",
    "what sensors should be added": "Useful additions are multi-depth soil moisture sensors, weather sensors such as humidity and wind speed, a camera for plant condition, and irrigation flow sensing once controlled irrigation is implemented.",
}

def answer_prompt(prompt: str) -> str:
    q = prompt.strip().lower()

    if not q:
        return "Ask me about the dashboard, a graph, the prediction tool, anomalies, or the sensors that could be added."

    # Figure matching
    if any(x in q for x in ["fig 1", "figure 1", "graph 1", "daily average soil moisture"]):
        return FIGURE_EXPLANATIONS["fig1"]
    if any(x in q for x in ["fig 2", "figure 2", "graph 2", "boxplot", "soil moisture distribution"]):
        return FIGURE_EXPLANATIONS["fig2"]
    if any(x in q for x in ["fig 3", "figure 3", "graph 3", "daily total rainfall"]):
        return FIGURE_EXPLANATIONS["fig3"]
    if any(x in q for x in ["fig 4", "figure 4", "graph 4", "overlay", "moisture and rainfall"]):
        return FIGURE_EXPLANATIONS["fig4"]
    if any(x in q for x in ["fig 5", "figure 5", "graph 5", "ec"]):
        return FIGURE_EXPLANATIONS["fig5"]

    # Prediction / anomaly / FAQ
    for key, value in FAQ.items():
        if key in q:
            return value

    if any(x in q for x in ["predict", "prediction", "next hour"]):
        return FAQ["what does the prediction mean"]

    if any(x in q for x in ["anomaly", "abnormal", "outlier", "weird", "unusual"]):
        return FAQ["what are the anomalies"]

    if any(x in q for x in ["sensor", "camera", "add", "improve"]):
        return FAQ["what sensors should be added"]

    if any(x in q for x in ["dashboard", "what does this do", "purpose"]):
        return FAQ["what does this dashboard do"]

    if any(x in q for x in ["irrigation zero", "why no irrigation", "irrigation"]):
        return FAQ["why is irrigation zero"]

    return (
        "I can help explain the graphs, the prediction tool, anomalies, and sensor suggestions. "
        "Try asking: 'Explain Figure 4', 'What does the EC graph mean?', or 'What anomalies do you see?'"
    )

# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.markdown("## 🌱 Dashboard Filters")
plot_options = ["All"] + sorted(df["plot_id"].dropna().unique().tolist())
selected_plot = st.sidebar.selectbox("Select Plot", plot_options)

date_min = df["timestamp"].min().date()
date_max = df["timestamp"].max().date()
selected_dates = st.sidebar.date_input(
    "Select Date Range",
    value=(date_min, date_max),
    min_value=date_min,
    max_value=date_max
)

if isinstance(selected_dates, tuple) and len(selected_dates) == 2:
    start_date, end_date = selected_dates
else:
    start_date, end_date = date_min, date_max

filtered_df = df[
    (df["timestamp"].dt.date >= start_date) &
    (df["timestamp"].dt.date <= end_date)
].copy()

if selected_plot != "All":
    filtered_df = filtered_df[filtered_df["plot_id"] == selected_plot]

daily = filtered_df.groupby(["date", "plot_id"], as_index=False).agg(
    daily_moisture=("soil_moisture_pct", "mean"),
    daily_rainfall=("rainfall_mm", "sum"),
    daily_ec=("soil_ec_ds_m", "mean"),
    daily_temp=("soil_temp_c", "mean"),
    daily_ph=("soil_ph", "mean")
)

# =========================================================
# HEADER
# =========================================================
st.markdown('<div class="main-title">🌿 Plantation Monitoring Dashboard</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Interactive monitoring of soil moisture, rainfall, EC, temperature, pH, and simple predictive insights for irrigation-related analysis.</div>',
    unsafe_allow_html=True
)

# =========================================================
# KPIs + ML
# =========================================================
avg_moisture = filtered_df["soil_moisture_pct"].mean()
avg_temp = filtered_df["soil_temp_c"].mean()
avg_ec = filtered_df["soil_ec_ds_m"].mean()
avg_ph = filtered_df["soil_ph"].mean()
total_rainfall = filtered_df["rainfall_mm"].sum()
total_irrigation = filtered_df["irrigation_mm"].sum()

k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Avg Moisture (%)", f"{avg_moisture:.2f}")
k2.metric("Avg Temp (°C)", f"{avg_temp:.2f}")
k3.metric("Avg EC (dS/m)", f"{avg_ec:.2f}")
k4.metric("Avg pH", f"{avg_ph:.2f}")
k5.metric("Total Rainfall (mm)", f"{total_rainfall:.0f}")
k6.metric("Total Irrigation (mm)", f"{total_irrigation:.0f}")

st.markdown(
    """
    <div class="insight-box">
    <b>Key insight:</b> Soil moisture does not increase proportionally with rainfall, suggesting that
    soil retention, drainage, and delayed absorption play important roles in determining water availability.
    </div>
    """,
    unsafe_allow_html=True
)

# =========================================================
# AI ASSISTANT + ML PANEL
# =========================================================
left, right = st.columns([1.1, 1])

with left:
    st.markdown('<div class="section-title">🌱 Sprout AI Assistant</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="sprout-card">
        <div style="display:flex; gap:14px; align-items:center;">
            <div>
                <svg width="88" height="88" viewBox="0 0 120 120" xmlns="http://www.w3.org/2000/svg">
                  <ellipse cx="60" cy="82" rx="26" ry="22" fill="#9b6b43"/>
                  <ellipse cx="60" cy="78" rx="22" ry="18" fill="#b77c4f"/>
                  <path d="M58 68 C56 56, 56 48, 60 40 C64 48, 64 56, 62 68" fill="#4c9a49"/>
                  <path d="M60 42 C45 36, 38 25, 40 15 C54 16, 62 25, 66 35" fill="#7fcf62"/>
                  <path d="M60 42 C75 36, 82 25, 80 15 C66 16, 58 25, 54 35" fill="#69b850"/>
                  <circle cx="52" cy="76" r="3.2" fill="#2a2a2a"/>
                  <circle cx="68" cy="76" r="3.2" fill="#2a2a2a"/>
                  <path d="M52 86 Q60 92 68 86" stroke="#2a2a2a" stroke-width="3" fill="none" stroke-linecap="round"/>
                  <circle cx="47" cy="82" r="2.8" fill="#f6b6a5"/>
                  <circle cx="73" cy="82" r="2.8" fill="#f6b6a5"/>
                </svg>
            </div>
            <div>
                <div class="sprout-title">Sprout Buddy</div>
                <div class="sprout-subtitle">Ask me about figures, anomalies, prediction, or dashboard insights.</div>
                <div class="faq-chip">Explain Figure 1</div>
                <div class="faq-chip">What anomalies do you see?</div>
                <div class="faq-chip">What does the prediction mean?</div>
                <div class="faq-chip">What sensors should be added?</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            ("bot", "Hi! I’m Sprout Buddy 🌱 Ask me to explain any figure or dashboard insight.")
        ]

    for speaker, text in st.session_state.chat_history:
        css_class = "chat-bubble-bot" if speaker == "bot" else "chat-bubble-user"
        st.markdown(f'<div class="{css_class}">{text}</div>', unsafe_allow_html=True)

    with st.form("chat_form", clear_on_submit=True):
        user_prompt = st.text_input("Ask Sprout Buddy", placeholder="e.g. Explain Figure 4")
        submitted = st.form_submit_button("Send")

    quick_cols = st.columns(4)
    quick_prompts = ["Explain Figure 1", "Explain Figure 4", "What anomalies do you see?", "What sensors should be added?"]
    for col, qp in zip(quick_cols, quick_prompts):
        if col.button(qp):
            st.session_state.chat_history.append(("user", qp))
            st.session_state.chat_history.append(("bot", answer_prompt(qp)))
            st.rerun()

    if submitted and user_prompt:
        st.session_state.chat_history.append(("user", user_prompt))
        st.session_state.chat_history.append(("bot", answer_prompt(user_prompt)))
        st.rerun()

with right:
    st.markdown('<div class="section-title">🤖 Next-Hour Moisture Prediction & Risk Panel</div>', unsafe_allow_html=True)

    latest_by_plot = filtered_df.sort_values("timestamp").groupby("plot_id").tail(1).copy()

    pred_rows = []
    for _, row in latest_by_plot.iterrows():
        pred = predict_next_moisture(row)
        risk_flags = []
        if pred < 18:
            risk_flags.append("Low predicted moisture")
        if row["soil_ec_ds_m"] > 2.5:
            risk_flags.append("High EC")
        if row["rainfall_mm"] == 0 and pred < 20:
            risk_flags.append("Dry/no-rain condition")

        pred_rows.append({
            "plot_id": row["plot_id"],
            "latest_time": row["timestamp"],
            "current_moisture": round(float(row["soil_moisture_pct"]), 2),
            "predicted_next_hour_moisture": round(pred, 2),
            "current_ec": round(float(row["soil_ec_ds_m"]), 2),
            "risk_status": " / ".join(risk_flags) if risk_flags else "Low risk"
        })

    pred_df = pd.DataFrame(pred_rows).sort_values("plot_id")
    st.dataframe(pred_df, use_container_width=True, hide_index=True)

    if selected_plot == "All":
        risk_plot = st.selectbox("Choose plot for prediction spotlight", sorted(pred_df["plot_id"].unique()), key="risk_plot")
    else:
        risk_plot = selected_plot

    spotlight = pred_df[pred_df["plot_id"] == risk_plot].iloc[0]
    c1, c2, c3 = st.columns(3)
    c1.metric("Current Moisture", f'{spotlight["current_moisture"]:.2f}%')
    c2.metric("Predicted Next Hour", f'{spotlight["predicted_next_hour_moisture"]:.2f}%')
    delta_val = spotlight["predicted_next_hour_moisture"] - spotlight["current_moisture"]
    c3.metric("Predicted Change", f"{delta_val:+.2f}%")

    st.markdown(
        f"""
        <div class="insight-box">
        <b>{risk_plot} status:</b> {spotlight["risk_status"]}. The prediction is a lightweight prototype based on current soil and rainfall conditions.
        </div>
        """,
        unsafe_allow_html=True
    )

# =========================================================
# TABS
# =========================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "Trends", "Soil Quality", "Data Table", "Future Features"])

with tab1:
    st.markdown('<div class="section-title">Overview of Moisture, Rainfall, and Variability</div>', unsafe_allow_html=True)
    col1, col2 = st.columns([1.3, 1])

    with col1:
        fig_daily_moisture = px.line(
            daily, x="date", y="daily_moisture",
            color="plot_id" if selected_plot == "All" else None,
            markers=True
        )
        fig_daily_moisture.update_traces(line=dict(width=3))
        fig_daily_moisture.update_layout(
            title="Figure 1: Daily Average Soil Moisture",
            xaxis_title="Date", yaxis_title="Daily Avg Moisture (%)",
            plot_bgcolor="white", paper_bgcolor="white",
            legend_title_text="Plot", margin=dict(l=20, r=20, t=60, b=20)
        )
        st.plotly_chart(fig_daily_moisture, use_container_width=True)

    with col2:
        fig_box = px.box(
            filtered_df, x="plot_id", y="soil_moisture_pct", points="outliers"
        )
        fig_box.update_layout(
            title="Figure 2: Soil Moisture Distribution",
            xaxis_title="Plot", yaxis_title="Soil Moisture (%)",
            plot_bgcolor="white", paper_bgcolor="white",
            showlegend=False, margin=dict(l=20, r=20, t=60, b=20)
        )
        st.plotly_chart(fig_box, use_container_width=True)

    st.markdown('<div class="small-note">Daily averages reduce hourly noise and make plot-level comparisons easier to interpret.</div>', unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="section-title">Rainfall and Moisture Trends</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        fig_rain = px.line(
            daily, x="date", y="daily_rainfall",
            color="plot_id" if selected_plot == "All" else None,
            markers=True
        )
        fig_rain.update_traces(line=dict(width=3))
        fig_rain.update_layout(
            title="Figure 3: Daily Total Rainfall",
            xaxis_title="Date", yaxis_title="Rainfall (mm)",
            plot_bgcolor="white", paper_bgcolor="white",
            legend_title_text="Plot", margin=dict(l=20, r=20, t=60, b=20)
        )
        st.plotly_chart(fig_rain, use_container_width=True)

    with col2:
        overlay_plot = selected_plot
        if overlay_plot == "All":
            overlay_plot = st.selectbox("Choose one plot for overlay view", sorted(df["plot_id"].unique()), key="overlay_plot")

        overlay_df = df[
            (df["plot_id"] == overlay_plot) &
            (df["timestamp"].dt.date >= start_date) &
            (df["timestamp"].dt.date <= end_date)
        ].copy()

        overlay_daily = overlay_df.groupby("date", as_index=False).agg(
            daily_moisture=("soil_moisture_pct", "mean"),
            daily_rainfall=("rainfall_mm", "sum")
        )

        fig_overlay = go.Figure()
        fig_overlay.add_trace(go.Bar(
            x=overlay_daily["date"],
            y=overlay_daily["daily_rainfall"],
            name="Daily Rainfall (mm)",
            opacity=0.35
        ))
        fig_overlay.add_trace(go.Scatter(
            x=overlay_daily["date"],
            y=overlay_daily["daily_moisture"],
            mode="lines+markers",
            name="Daily Avg Soil Moisture (%)",
            yaxis="y2",
            line=dict(width=3)
        ))
        fig_overlay.update_layout(
            title=f"Figure 4: {overlay_plot} Moisture and Rainfall Overlay",
            xaxis=dict(title="Date"),
            yaxis=dict(title="Daily Rainfall (mm)"),
            yaxis2=dict(title="Daily Avg Soil Moisture (%)", overlaying="y", side="right"),
            plot_bgcolor="white", paper_bgcolor="white",
            legend=dict(orientation="h"), margin=dict(l=20, r=20, t=60, b=20)
        )
        st.plotly_chart(fig_overlay, use_container_width=True)

with tab3:
    st.markdown('<div class="section-title">EC, Temperature, and pH Conditions</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)

    with col1:
        fig_ec = px.line(
            daily, x="date", y="daily_ec",
            color="plot_id" if selected_plot == "All" else None,
            markers=True
        )
        fig_ec.update_traces(line=dict(width=3))
        fig_ec.update_layout(
            title="Figure 5: Daily Average Soil EC",
            xaxis_title="Date", yaxis_title="EC (dS/m)",
            plot_bgcolor="white", paper_bgcolor="white",
            legend_title_text="Plot", margin=dict(l=20, r=20, t=60, b=20)
        )
        st.plotly_chart(fig_ec, use_container_width=True)

    with col2:
        fig_temp = px.line(
            daily, x="date", y="daily_temp",
            color="plot_id" if selected_plot == "All" else None,
            markers=True
        )
        fig_temp.update_traces(line=dict(width=3))
        fig_temp.update_layout(
            title="Daily Average Soil Temperature",
            xaxis_title="Date", yaxis_title="Temperature (°C)",
            plot_bgcolor="white", paper_bgcolor="white",
            legend_title_text="Plot", margin=dict(l=20, r=20, t=60, b=20)
        )
        st.plotly_chart(fig_temp, use_container_width=True)

    with col3:
        fig_ph = px.line(
            daily, x="date", y="daily_ph",
            color="plot_id" if selected_plot == "All" else None,
            markers=True
        )
        fig_ph.update_traces(line=dict(width=3))
        fig_ph.update_layout(
            title="Daily Average Soil pH",
            xaxis_title="Date", yaxis_title="pH",
            plot_bgcolor="white", paper_bgcolor="white",
            legend_title_text="Plot", margin=dict(l=20, r=20, t=60, b=20)
        )
        st.plotly_chart(fig_ph, use_container_width=True)

with tab4:
    st.markdown('<div class="section-title">Filtered Dataset</div>', unsafe_allow_html=True)
    st.dataframe(filtered_df, use_container_width=True)

    summary_by_plot = filtered_df.groupby("plot_id").agg(
        avg_soil_moisture_pct=("soil_moisture_pct", "mean"),
        std_soil_moisture_pct=("soil_moisture_pct", "std"),
        min_soil_moisture_pct=("soil_moisture_pct", "min"),
        max_soil_moisture_pct=("soil_moisture_pct", "max"),
        avg_soil_temp_c=("soil_temp_c", "mean"),
        avg_soil_ec_ds_m=("soil_ec_ds_m", "mean"),
        avg_soil_ph=("soil_ph", "mean"),
        total_rainfall_mm=("rainfall_mm", "sum"),
        total_irrigation_mm=("irrigation_mm", "sum")
    ).round(2)

    st.markdown('<div class="section-title">Summary Statistics by Plot</div>', unsafe_allow_html=True)
    st.dataframe(summary_by_plot, use_container_width=True)


with tab5:
    st.markdown('<div class="section-title">Future System Upgrades</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="sprout-card">
            <div class="sprout-title">Why this tab exists</div>
            <div class="sprout-subtitle">
                The current system measures soil state variables well, but it does not directly observe infiltration delay,
                evapotranspiration, or crop stress. The following upgrades would improve interpretability and irrigation decision-making.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    c1, c2 = st.columns(2)

    with c1:
        st.markdown(
            """
            <div class="sprout-card">
                <div class="sprout-title">Multi-depth Soil Moisture</div>
                <div class="sprout-subtitle">
                    Add moisture sensors at multiple depths (for example 10 cm, 20 cm, and 30 cm) to track how rainfall moves vertically through the soil.
                    This helps determine whether water only wets the surface or actually reaches the crop root zone.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        depth_df = pd.DataFrame({
            "Depth": ["10 cm", "20 cm", "30 cm"],
            "Example Moisture (%)": [29, 23, 18]
        })
        fig_depth = px.bar(
            depth_df,
            x="Depth",
            y="Example Moisture (%)",
            title="Conceptual Multi-Depth Soil Moisture View"
        )
        fig_depth.update_layout(
            plot_bgcolor="white",
            paper_bgcolor="white",
            margin=dict(l=20, r=20, t=60, b=20)
        )
        st.plotly_chart(fig_depth, use_container_width=True)

        st.markdown(
            """
            <div class="sprout-card">
                <div class="sprout-title">Distributed Plot Sensing</div>
                <div class="sprout-subtitle">
                    Add more sensors within each plot to distinguish true soil heterogeneity from isolated sensor readings.
                    This would support more reliable plot-specific decisions rather than a one-size-fits-all irrigation strategy.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:
        st.markdown(
            """
            <div class="sprout-card">
                <div class="sprout-title">Weather & Evapotranspiration</div>
                <div class="sprout-subtitle">
                    Add weather sensors such as air temperature, humidity, wind speed, and solar radiation to estimate evapotranspiration.
                    This helps explain why soil moisture may fall even when rainfall is absent or weakly correlated with moisture.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        weather_df = pd.DataFrame({
            "Variable": ["Air Temp", "Humidity", "Wind Speed", "Solar Radiation"],
            "Relative Importance": [0.75, 0.85, 0.55, 0.95]
        })
        fig_weather = px.bar(
            weather_df,
            x="Variable",
            y="Relative Importance",
            title="Conceptual Weather Inputs for ET Estimation"
        )
        fig_weather.update_layout(
            plot_bgcolor="white",
            paper_bgcolor="white",
            yaxis_title="Importance",
            margin=dict(l=20, r=20, t=60, b=20),
            showlegend=False
        )
        st.plotly_chart(fig_weather, use_container_width=True)

        st.markdown(
            """
            <div class="sprout-card">
                <div class="sprout-title">Camera / Crop Health Monitoring</div>
                <div class="sprout-subtitle">
                    Add a simple camera system to monitor leaf colour, wilting, and visible plant stress.
                    This connects soil conditions to actual crop response and provides a more practical indicator of irrigation effectiveness.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown('<div class="section-title">How these upgrades improve the dashboard</div>', unsafe_allow_html=True)

    improvement_df = pd.DataFrame({
        "Upgrade": [
            "Multi-depth sensing",
            "Weather / ET sensing",
            "Distributed spatial sensing",
            "Camera-based crop monitoring"
        ],
        "What it adds": [
            "Tracks infiltration and root-zone water availability",
            "Explains environmental moisture loss",
            "Improves cross-plot comparison and local diagnosis",
            "Links soil data to visible plant stress"
        ]
    })
    st.dataframe(improvement_df, use_container_width=True, hide_index=True)

    st.markdown(
        """
        <div class="insight-box">
        <b>Design impact:</b> These future sensing modules would transform the dashboard from a descriptive monitoring interface
        into a more diagnostic and predictive decision-support tool. They would improve observability of soil-water processes
        and make irrigation planning more reliable.
        </div>
        """,
        unsafe_allow_html=True
    )
