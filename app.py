import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Plantation Monitoring Dashboard", layout="wide")

# -----------------------------
# Load data
# -----------------------------
@st.cache_data
def load_data():
    file_path = "plantation_soil_data (2).xlsx"
    df = pd.read_excel(file_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], dayfirst=True, errors="coerce")
    df["date"] = df["timestamp"].dt.date
    return df.sort_values(["plot_id", "timestamp"]).reset_index(drop=True)

df = load_data()

st.title("Plantation / Agricultural Monitoring Dashboard")
st.markdown("Interactive dashboard for soil conditions, rainfall, and irrigation-related analysis.")

# -----------------------------
# Sidebar filters
# -----------------------------
st.sidebar.header("Filters")

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

# -----------------------------
# KPI cards
# -----------------------------
avg_moisture = filtered_df["soil_moisture_pct"].mean()
avg_temp = filtered_df["soil_temp_c"].mean()
avg_ec = filtered_df["soil_ec_ds_m"].mean()
avg_ph = filtered_df["soil_ph"].mean()
total_rainfall = filtered_df["rainfall_mm"].sum()
total_irrigation = filtered_df["irrigation_mm"].sum()

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Avg Moisture (%)", f"{avg_moisture:.2f}")
c2.metric("Avg Temp (°C)", f"{avg_temp:.2f}")
c3.metric("Avg EC (dS/m)", f"{avg_ec:.2f}")
c4.metric("Avg pH", f"{avg_ph:.2f}")
c5.metric("Total Rainfall (mm)", f"{total_rainfall:.0f}")
c6.metric("Total Irrigation (mm)", f"{total_irrigation:.0f}")

st.markdown("---")

# -----------------------------
# Chart 1: Soil Moisture vs Time
# -----------------------------
st.subheader("Soil Moisture vs Time")

fig_moisture = px.line(
    filtered_df,
    x="timestamp",
    y="soil_moisture_pct",
    color="plot_id" if selected_plot == "All" else None,
    title="Soil Moisture Over Time"
)
fig_moisture.update_layout(xaxis_title="Time", yaxis_title="Soil Moisture (%)")
st.plotly_chart(fig_moisture, use_container_width=True)

# -----------------------------
# Chart 2: Rainfall vs Time
# -----------------------------
st.subheader("Rainfall vs Time")

rainfall_time = filtered_df.groupby(["timestamp", "plot_id"], as_index=False)["rainfall_mm"].sum()

if selected_plot == "All":
    fig_rain = px.bar(
        rainfall_time,
        x="timestamp",
        y="rainfall_mm",
        color="plot_id",
        title="Rainfall Over Time"
    )
else:
    fig_rain = px.bar(
        rainfall_time,
        x="timestamp",
        y="rainfall_mm",
        title=f"Rainfall Over Time - {selected_plot}"
    )

fig_rain.update_layout(xaxis_title="Time", yaxis_title="Rainfall (mm)")
st.plotly_chart(fig_rain, use_container_width=True)

# -----------------------------
# Daily aggregation
# -----------------------------
daily = filtered_df.groupby(["date", "plot_id"], as_index=False).agg(
    daily_moisture=("soil_moisture_pct", "mean"),
    daily_rainfall=("rainfall_mm", "sum"),
    daily_ec=("soil_ec_ds_m", "mean")
)

# -----------------------------
# Chart 3 and 4 side by side
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("Daily Average Soil Moisture")
    fig_daily_moisture = px.line(
        daily,
        x="date",
        y="daily_moisture",
        color="plot_id" if selected_plot == "All" else None,
        markers=True,
        title="Daily Average Soil Moisture"
    )
    fig_daily_moisture.update_layout(xaxis_title="Date", yaxis_title="Daily Avg Moisture (%)")
    st.plotly_chart(fig_daily_moisture, use_container_width=True)

with col2:
    st.subheader("Daily Average Soil EC")
    fig_daily_ec = px.line(
        daily,
        x="date",
        y="daily_ec",
        color="plot_id" if selected_plot == "All" else None,
        markers=True,
        title="Daily Average Soil EC"
    )
    fig_daily_ec.update_layout(xaxis_title="Date", yaxis_title="Daily Avg EC (dS/m)")
    st.plotly_chart(fig_daily_ec, use_container_width=True)

# -----------------------------
# Chart 5: Soil Moisture Distribution
# -----------------------------
st.subheader("Soil Moisture Distribution")

fig_box = px.box(
    filtered_df,
    x="plot_id",
    y="soil_moisture_pct",
    title="Soil Moisture Distribution by Plot"
)
fig_box.update_layout(xaxis_title="Plot", yaxis_title="Soil Moisture (%)")
st.plotly_chart(fig_box, use_container_width=True)

# -----------------------------
# Chart 6: Daily Moisture + Rainfall Overlay
# -----------------------------
st.subheader("Daily Soil Moisture and Rainfall Overlay")

overlay_plot = selected_plot
if overlay_plot == "All":
    overlay_plot = st.selectbox("Choose one plot for overlay view", sorted(df["plot_id"].unique()))

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
    opacity=0.4,
    yaxis="y2"
))
fig_overlay.add_trace(go.Scatter(
    x=overlay_daily["date"],
    y=overlay_daily["daily_moisture"],
    mode="lines+markers",
    name="Daily Avg Soil Moisture (%)"
))

fig_overlay.update_layout(
    title=f"{overlay_plot}: Daily Soil Moisture and Rainfall",
    xaxis=dict(title="Date"),
    yaxis=dict(title="Daily Avg Soil Moisture (%)"),
    yaxis2=dict(
        title="Daily Rainfall (mm)",
        overlaying="y",
        side="right"
    ),
    legend=dict(orientation="h")
)

st.plotly_chart(fig_overlay, use_container_width=True)

# -----------------------------
# Data table
# -----------------------------
st.subheader("Filtered Data Table")
st.dataframe(filtered_df, use_container_width=True)

# -----------------------------
# Insight box
# -----------------------------
st.markdown("---")
st.info(
    "Key insight: rainfall does not always lead to a proportional increase in soil moisture, "
    "suggesting that drainage, retention, and delayed absorption influence soil behaviour."
)
