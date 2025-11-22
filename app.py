"""
Inflation and Household Consumption in Canada: A Comparative OECD Analysis Dashboard
Author: Aishwarya & Scarlett
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

# Page Configuration
st.set_page_config(
    page_title="Inflation & Household Consumption Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {background-color: #f5f7fa;}
    .stMetric {
        background-color: white; 
        padding: 15px; 
        border-radius: 10px; 
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stMetric label {color: #2c3e50 !important; font-weight: 600 !important;}
    .stMetric [data-testid="stMetricValue"] {color: #1f77b4 !important; font-size: 2rem !important; font-weight: bold !important;}
    h1 {color: #1f77b4; font-size: 2.5rem;}
    h2 {color: #2c3e50; margin-top: 20px;}
    .insight-box {
        background-color: #e8f4f8;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 15px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("📊 Inflation and Household Consumption in Canada")
st.markdown("### A Comparative OECD Analysis (2020–2024)")
st.markdown("---")

# File URLs
CPI_URL = "https://github.com/aishwaryashan20/dashboard_creation/releases/download/v1.0/Monthly.Consumer.Price.Indices.CPI.HICP.csv"
CONS_URL = "https://github.com/aishwaryashan20/dashboard_creation/releases/download/v1.0/Annual.Household.Final.Consumption.Expenditure.COICOP.csv"

@st.cache_data
def load_cpi_recent():
    """Load CPI data for 2020-2024"""
    df = pd.read_csv(CPI_URL, low_memory=False, on_bad_lines='skip')
    df['OBS_VALUE'] = pd.to_numeric(df['OBS_VALUE'], errors='coerce')
    df['TIME_PERIOD'] = pd.to_datetime(df['TIME_PERIOD'], errors='coerce')
    df = df[(df['TIME_PERIOD'].dt.year >= 2020) & (df['TIME_PERIOD'].dt.year <= 2024)].copy()
    return df

@st.cache_data  
def load_datasets_for_correlation():
    """Load full datasets to find overlapping years"""
    cpi = pd.read_csv(CPI_URL, low_memory=False, on_bad_lines='skip')
    cons = pd.read_csv(CONS_URL, low_memory=False, on_bad_lines='skip')
    
    # Extract years safely
    cpi['Year'] = pd.to_numeric(cpi['TIME_PERIOD'].astype(str).str[:4], errors='coerce')
    cons['Year'] = pd.to_numeric(cons['TIME_PERIOD'].astype(str).str[:4], errors='coerce')
    
    # Remove rows with invalid years
    cpi = cpi[cpi['Year'].notna()].copy()
    cons = cons[cons['Year'].notna()].copy()
    
    # Convert to int
    cpi['Year'] = cpi['Year'].astype(int)
    cons['Year'] = cons['Year'].astype(int)
    
    # Convert values
    cpi['OBS_VALUE'] = pd.to_numeric(cpi['OBS_VALUE'], errors='coerce')
    cons['OBS_VALUE'] = pd.to_numeric(cons['OBS_VALUE'], errors='coerce')
    
    return cpi, cons

# Load data
with st.spinner("🔄 Loading datasets..."):
    cpi_df = load_cpi_recent()
    st.sidebar.success(f"✅ Loaded {len(cpi_df):,} CPI records (2020-2024)")

# Sidebar
with st.sidebar:
    st.header("📋 Dashboard Guide")
    st.markdown("""
    **Three Key Insights:**
    1. 🎯 Country Clustering
    2. 📈 Inflation-Consumption Correlation
    3. 📅 Temporal Trends
    """)

# Overview
st.header("📊 Dataset Overview")
col1, col2 = st.columns(2)
with col1:
    st.metric("CPI Records (2020-2024)", f"{len(cpi_df):,}")
with col2:
    st.metric("Countries", cpi_df['REF_AREA'].nunique())

st.markdown("---")

# ====================
# INSIGHT 1: CLUSTERING
# ====================
st.markdown('<div class="insight-box">', unsafe_allow_html=True)
st.header("🎯 Insight 1: Country Clustering by Inflation Patterns")
st.markdown("Groups OECD countries based on their inflation trajectories since 2020.")
st.markdown('</div>', unsafe_allow_html=True)

cluster_data = cpi_df.groupby('REF_AREA')['OBS_VALUE'].agg([
    ('avg', 'mean'),
    ('volatility', 'std'),
    ('peak', 'max')
]).dropna()

if len(cluster_data) >= 3:
    scaler = StandardScaler()
    scaled = scaler.fit_transform(cluster_data)
    
    n_clusters = min(4, len(cluster_data))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_data['Group'] = ['Group ' + str(i+1) for i in kmeans.fit_predict(scaled)]
    
    col1, col2 = st.columns(2)
    
    with col1:
        counts = cluster_data['Group'].value_counts().reset_index()
        fig1 = px.bar(counts, x='Group', y='count', title='Countries per Cluster',
                     color='Group', color_discrete_sequence=px.colors.qualitative.Set2)
        fig1.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        fig2 = px.scatter(cluster_data.reset_index(), x='avg', y='volatility',
                         color='Group', size='peak', hover_data=['REF_AREA'],
                         title='Inflation: Average vs Volatility',
                         color_discrete_sequence=px.colors.qualitative.Set2)
        fig2.update_layout(height=400)
        st.plotly_chart(fig2, use_container_width=True)
    
    st.subheader("📋 Cluster Details")
    for group in sorted(cluster_data['Group'].unique()):
        countries = cluster_data[cluster_data['Group'] == group].index.tolist()
        is_canada = 'CAN' in countries
        with st.expander(f"{group} ({'🍁 CANADA' if is_canada else f'{len(countries)} countries'})", expanded=is_canada):
            st.write(", ".join(countries))

st.markdown("---")

# ====================
# INSIGHT 2: CORRELATION
# ====================
st.markdown('<div class="insight-box">', unsafe_allow_html=True)
st.header("📈 Insight 2: Inflation-Consumption Correlation")
st.markdown("Examines the relationship between inflation rates and household consumption using overlapping historical data.")
st.markdown('</div>', unsafe_allow_html=True)

with st.spinner("Loading full datasets for correlation analysis..."):
    cpi_full, cons_full = load_datasets_for_correlation()
    
    # Find overlapping years
    cpi_years = set(cpi_full['Year'].unique())
    cons_years = set(cons_full['Year'].unique())
    overlap = sorted(list(cpi_years & cons_years))
    
    st.info(f"📊 CPI: {min(cpi_years)}-{max(cpi_years)} | Consumption: {min(cons_years)}-{max(cons_years)} | **Overlap: {min(overlap)}-{max(overlap)}** ({len(overlap)} years)")

if len(overlap) >= 5:
    # Filter to overlapping years
    cpi_overlap = cpi_full[cpi_full['Year'].isin(overlap)].copy()
    cons_overlap = cons_full[cons_full['Year'].isin(overlap)].copy()
    
    # Aggregate
    cpi_agg = cpi_overlap.groupby(['REF_AREA', 'Year'])['OBS_VALUE'].mean().reset_index()
    cpi_agg.columns = ['Country', 'Year', 'Inflation']
    
    cons_agg = cons_overlap.groupby(['REF_AREA', 'Year'])['OBS_VALUE'].mean().reset_index()
    cons_agg.columns = ['Country', 'Year', 'Consumption']
    
    # Merge
    merged = pd.merge(cpi_agg, cons_agg, on=['Country', 'Year'], how='inner')
    
    st.success(f"✅ Merged {len(merged)} data points across {merged['Country'].nunique()} countries")
    
    if len(merged) > 20:
        # Clean outliers
        merged_clean = merged[
            (merged['Inflation'] < merged['Inflation'].quantile(0.98)) &
            (merged['Consumption'] < merged['Consumption'].quantile(0.98))
        ]
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig3 = px.scatter(merged_clean, x='Inflation', y='Consumption',
                            color='Country', hover_data=['Year'],
                            title=f'Inflation vs Consumption ({min(overlap)}-{max(overlap)})')
            fig3.update_layout(height=450, showlegend=False)
            st.plotly_chart(fig3, use_container_width=True)
        
        with col2:
            canada = merged[merged['Country'] == 'CAN']
            if len(canada) > 0:
                fig4 = go.Figure()
                fig4.add_trace(go.Scatter(x=canada['Year'], y=canada['Inflation'],
                                         name='Inflation', line=dict(color='#e74c3c', width=3)))
                cons_norm = (canada['Consumption'] / canada['Consumption'].max()) * canada['Inflation'].max()
                fig4.add_trace(go.Scatter(x=canada['Year'], y=cons_norm,
                                         name='Consumption', line=dict(color='#3498db', width=3, dash='dash')))
                fig4.update_layout(title='🍁 Canada Trends', height=450)
                st.plotly_chart(fig4, use_container_width=True)
            else:
                fig4 = px.box(merged, x='Year', y='Inflation', title='Inflation by Year')
                fig4.update_layout(height=450)
                st.plotly_chart(fig4, use_container_width=True)
        
        # Correlation
        corr = merged['Inflation'].corr(merged['Consumption'])
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Correlation", f"{corr:.3f}")
        with col2:
            strength = "Strong" if abs(corr) > 0.5 else "Moderate" if abs(corr) > 0.3 else "Weak"
            st.metric("Strength", strength)
        with col3:
            direction = "Positive ↗" if corr > 0 else "Negative ↘"
            st.metric("Direction", direction)
        
        st.subheader("📊 Interpretation")
        if corr > 0.5:
            st.success(f"**Strong Positive Correlation** (r={corr:.3f}): Higher inflation associated with higher consumption during {min(overlap)}-{max(overlap)}. Households maintained spending despite price increases.")
        elif corr > 0.3:
            st.info(f"**Moderate Correlation** (r={corr:.3f}): Some relationship exists, but other factors also influence consumption patterns.")
        elif corr < -0.3:
            st.warning(f"**Negative Correlation** (r={corr:.3f}): Higher inflation reduced consumption, indicating price sensitivity.")
        else:
            st.info(f"**Weak Correlation** (r={corr:.3f}): Consumption driven by factors beyond inflation rates.")
    else:
        st.warning("Insufficient merged data for detailed analysis")
else:
    st.error("Insufficient overlapping years for correlation analysis")

st.markdown("---")

# ====================
# INSIGHT 3: TEMPORAL
# ====================
st.markdown('<div class="insight-box">', unsafe_allow_html=True)
st.header("📅 Insight 3: Inflation Evolution (2020-2024)")
st.markdown("Tracks inflation trends, identifying peaks and patterns across OECD countries.")
st.markdown('</div>', unsafe_allow_html=True)

cpi_ts = cpi_df.sort_values('TIME_PERIOD')
key_countries = ['CAN', 'USA', 'GBR', 'DEU', 'FRA', 'JPN']
available = [c for c in key_countries if c in cpi_ts['REF_AREA'].unique()]
cpi_plot = cpi_ts[cpi_ts['REF_AREA'].isin(available)]

fig5 = px.line(cpi_plot, x='TIME_PERIOD', y='OBS_VALUE', color='REF_AREA',
              title='Monthly Inflation Trends: Canada vs Key OECD Countries',
              labels={'OBS_VALUE': 'Inflation Rate (%)', 'TIME_PERIOD': 'Date'},
              color_discrete_sequence=px.colors.qualitative.Bold)

for trace in fig5.data:
    if 'CAN' in trace.name:
        trace.update(line=dict(width=4))

fig5.update_layout(height=500, hovermode='x unified')
st.plotly_chart(fig5, use_container_width=True)

col1, col2, col3, col4 = st.columns(4)
with col1:
    can_avg = cpi_plot[cpi_plot['REF_AREA'] == 'CAN']['OBS_VALUE'].mean()
    st.metric("🍁 Canada Avg", f"{can_avg:.2f}%")
with col2:
    oecd_avg = cpi_plot['OBS_VALUE'].mean()
    st.metric("OECD Avg", f"{oecd_avg:.2f}%")
with col3:
    peak = cpi_plot['OBS_VALUE'].max()
    st.metric("Peak", f"{peak:.2f}%")
with col4:
    recent = cpi_plot[cpi_plot['TIME_PERIOD'] >= '2024-01-01']['OBS_VALUE'].mean()
    st.metric("2024 Avg", f"{recent:.2f}%")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #7f8c8d; padding: 20px;'>
<p><strong>Inflation and Household Consumption Analysis Dashboard</strong></p>
<p>Data: OECD | Period: 2020-2024 | Created by Aishwaya & Scarlett for ALY6110</p>
</div>
""", unsafe_allow_html=True)



