"""
Inflation and Household Consumption in Canada: A Comparative OECD Analysis Dashboard
Author: Aishu
Course: ALY6080 - Integrated Experiential Learning
Dataset: OECD CPI and Household Consumption Data (2020-2024)
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from scipy import stats
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
    .stMetric label {
        color: #2c3e50 !important;
        font-weight: 600 !important;
    }
    .stMetric [data-testid="stMetricValue"] {
        color: #1f77b4 !important;
        font-size: 2rem !important;
        font-weight: bold !important;
    }
    .stMetric [data-testid="stMetricDelta"] {
        color: #27ae60 !important;
    }
    h1 {color: #1f77b4; font-size: 2.5rem;}
    h2 {color: #2c3e50; margin-top: 20px;}
    h3 {color: #34495e;}
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

# File paths - GitHub Release URLs
USE_URLS = True

if USE_URLS:
    CPI_FILE = "https://github.com/aishwaryashan20/dashboard_creation/releases/download/v1.0/Monthly.Consumer.Price.Indices.CPI.HICP.csv"
    CONSUMPTION_FILE = "https://github.com/aishwaryashan20/dashboard_creation/releases/download/v1.0/Annual.Household.Final.Consumption.Expenditure.COICOP.csv"
else:
    CPI_FILE = "Monthly Consumer Price Indices (CPI, HICP).csv"
    CONSUMPTION_FILE = "Annual Household Final Consumption Expenditure (COICOP).csv"

@st.cache_data
def load_and_process_cpi(file_path):
    """Load and automatically process CPI dataset"""
    try:
        if file_path.startswith('http'):
            st.sidebar.info("📥 Downloading CPI data from GitHub...")
            df = pd.read_csv(file_path, low_memory=False, on_bad_lines='skip', encoding='utf-8')
        else:
            df = pd.read_csv(file_path, low_memory=False, on_bad_lines='skip')
        
        st.sidebar.info(f"📂 CPI file loaded: {len(df):,} total records")
        
        with st.sidebar.expander("🔍 Debug: View CPI Columns"):
            st.write("**Available columns:**", df.columns.tolist())
            st.dataframe(df.head(3))
        
        # Auto-detect columns
        country_col = time_col = value_col = None
        
        for col in ['REF_AREA', 'LOCATION', 'Country']:
            if col in df.columns:
                country_col = col
                break
        
        for col in ['TIME_PERIOD', 'Time', 'TIME']:
            if col in df.columns:
                time_col = col
                break
        
        for col in ['OBS_VALUE', 'Value', 'VALUE']:
            if col in df.columns:
                value_col = col
                break
        
        if not all([country_col, time_col, value_col]):
            for col in df.columns:
                if 'AREA' in col.upper() or 'COUNTRY' in col.upper():
                    country_col = col
                elif 'TIME' in col.upper() or 'PERIOD' in col.upper():
                    time_col = col
                elif 'VALUE' in col.upper() or 'OBS' in col.upper():
                    value_col = col
        
        st.sidebar.success(f"✅ CPI Columns:\n- Country: {country_col}\n- Time: {time_col}\n- Value: {value_col}")
        
        df[value_col] = pd.to_numeric(df[value_col], errors='coerce')
        df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
        
        valid_dates = df[time_col].notna().sum()
        st.sidebar.info(f"📅 Valid dates: {valid_dates:,}")
        
        df_filtered = df[
            (df[time_col].notna()) & 
            (df[time_col].dt.year >= 2020) & 
            (df[time_col].dt.year <= 2024)
        ].copy()
        
        st.sidebar.success(f"✅ CPI Data: {len(df_filtered):,} records (2020-2024)")
        return df_filtered, country_col, time_col, value_col
        
    except Exception as e:
        st.sidebar.error(f"❌ Error loading CPI: {str(e)}")
        return None, None, None, None

@st.cache_data
def load_and_process_consumption(file_path):
    """Load and automatically process Consumption dataset"""
    try:
        if file_path.startswith('http'):
            st.sidebar.info("📥 Downloading Consumption data...")
            df = pd.read_csv(file_path, low_memory=False, on_bad_lines='skip', encoding='utf-8')
        else:
            df = pd.read_csv(file_path, low_memory=False, on_bad_lines='skip')
        
        st.sidebar.info(f"📂 Consumption file loaded: {len(df):,} total records")
        
        # Auto-detect columns
        country_col = time_col = value_col = None
        
        for col in ['REF_AREA', 'LOCATION', 'Country']:
            if col in df.columns:
                country_col = col
                break
        
        for col in ['TIME_PERIOD', 'Time', 'TIME']:
            if col in df.columns:
                time_col = col
                break
        
        for col in ['OBS_VALUE', 'Value', 'VALUE']:
            if col in df.columns:
                value_col = col
                break
        
        if not all([country_col, time_col, value_col]):
            st.sidebar.error("❌ Could not detect consumption columns")
            return None, None, None, None
        
        df[value_col] = pd.to_numeric(df[value_col], errors='coerce')
        df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
        
        # Check year range
        valid_dates = df[time_col].notna()
        years = df[valid_dates][time_col].dt.year.unique()
        year_min = years.min() if len(years) > 0 else None
        year_max = years.max() if len(years) > 0 else None
        
        st.sidebar.warning(f"⚠️ Consumption Data: {year_min} to {year_max}")
        
        if year_max and year_max < 2020:
            st.sidebar.error(f"❌ No 2020-2024 data! Latest: {year_max}")
            return None, country_col, time_col, value_col
        
        df_filtered = df[
            (df[time_col].notna()) & 
            (df[time_col].dt.year >= 2020) & 
            (df[time_col].dt.year <= 2024)
        ].copy()
        
        st.sidebar.success(f"✅ Consumption: {len(df_filtered):,} records (2020-2024)")
        return df_filtered, country_col, time_col, value_col
        
    except Exception as e:
        st.sidebar.error(f"❌ Error loading Consumption: {str(e)}")
        return None, None, None, None

# Load datasets
with st.spinner("🔄 Loading OECD datasets..."):
    cpi_df, cpi_country, cpi_time, cpi_value = load_and_process_cpi(CPI_FILE)
    cons_df, cons_country, cons_time, cons_value = load_and_process_consumption(CONSUMPTION_FILE)

if cpi_df is None:
    st.error("⚠️ Could not load CPI dataset")
    st.stop()

# Sidebar
with st.sidebar:
    st.header("📋 Dashboard Guide")
    st.markdown("""
    **Three Key Insights:**
    1. 🎯 Country Clustering
    2. 📈 Inflation Analysis  
    3. 📅 Temporal Trends
    
    **Focus:** Canada vs OECD
    """)

# Dataset Overview
st.header("📊 Dataset Overview (2020-2024)")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("CPI Records", f"{len(cpi_df):,}")
with col2:
    cons_count = len(cons_df) if cons_df is not None else 0
    st.metric("Consumption Records", f"{cons_count:,}")
with col3:
    st.metric("Countries (CPI)", cpi_df[cpi_country].nunique())
with col4:
    cons_countries = cons_df[cons_country].nunique() if cons_df is not None else 0
    st.metric("Countries (Consumption)", cons_countries)

st.markdown("---")

# INSIGHT 1: Clustering
st.markdown('<div class="insight-box">', unsafe_allow_html=True)
st.header("🎯 Insight 1: Country Clustering by Inflation Patterns")
st.markdown("""
Groups OECD countries based on inflation trajectories since 2020.
Countries with similar patterns are clustered together.
""")
st.markdown('</div>', unsafe_allow_html=True)

try:
    cluster_data = cpi_df.groupby(cpi_country)[cpi_value].agg([
        ('avg_inflation', 'mean'),
        ('volatility', 'std'),
        ('peak_inflation', 'max')
    ]).dropna()
    
    if len(cluster_data) >= 3:
        scaler = StandardScaler()
        scaled = scaler.fit_transform(cluster_data)
        
        n_clusters = min(4, len(cluster_data))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_data['Cluster'] = kmeans.fit_predict(scaled)
        cluster_data['Group'] = ['Group ' + str(i+1) for i in cluster_data['Cluster']]
        
        col1, col2 = st.columns(2)
        
        with col1:
            counts = cluster_data['Group'].value_counts().reset_index()
            counts.columns = ['Group', 'Count']
            fig1 = px.bar(counts, x='Group', y='Count',
                         title='Countries per Cluster',
                         color='Group',
                         color_discrete_sequence=px.colors.qualitative.Set2)
            fig1.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            fig2 = px.scatter(cluster_data.reset_index(),
                            x='avg_inflation', y='volatility',
                            color='Group', size='peak_inflation',
                            hover_data=[cpi_country],
                            title='Inflation: Average vs Volatility',
                            color_discrete_sequence=px.colors.qualitative.Set2)
            fig2.update_layout(height=400)
            st.plotly_chart(fig2, use_container_width=True)
        
        st.subheader("📋 Cluster Details")
        for i in range(n_clusters):
            cluster_countries = cluster_data[cluster_data['Cluster'] == i]
            countries_list = cluster_countries.index.tolist()
            is_canada = any('CAN' in str(c) for c in countries_list)
            
            with st.expander(f"Group {i+1} ({'🍁 CANADA' if is_canada else f'{len(countries_list)} countries'})", expanded=is_canada):
                st.write(", ".join(countries_list))
except Exception as e:
    st.error(f"Error in clustering: {str(e)}")

st.markdown("---")

# INSIGHT 2: Correlation/Alternative Analysis
st.markdown('<div class="insight-box">', unsafe_allow_html=True)
st.header("📈 Insight 2: Inflation Analysis")
st.markdown("""
Analyzes inflation patterns and their economic implications.
""")
st.markdown('</div>', unsafe_allow_html=True)

if cons_df is None or len(cons_df) == 0:
    st.warning("⚠️ Consumption data not available for 2020-2024. Showing CPI-only analysis.")
    
    try:
        cpi_yearly = cpi_df.copy()
        cpi_yearly['Year'] = cpi_yearly[cpi_time].dt.year
        
        country_stats = cpi_yearly.groupby(cpi_country)[cpi_value].agg([
            ('avg', 'mean'),
            ('volatility', 'std'),
            ('peak', 'max')
        ]).reset_index()
        
        col1, col2 = st.columns(2)
        
        with col1:
            top10 = country_stats.nlargest(10, 'avg')
            fig = px.bar(top10, x=cpi_country, y='avg',
                        title='Top 10: Highest Average Inflation',
                        color='avg', color_continuous_scale='Reds')
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            volatile = country_stats.nlargest(10, 'volatility')
            fig = px.bar(volatile, x=cpi_country, y='volatility',
                        title='Top 10: Most Volatile Inflation',
                        color='volatility', color_continuous_scale='Blues')
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        # Canada metrics
        canada_data = country_stats[country_stats[cpi_country].str.contains('CAN', case=False, na=False)]
        if len(canada_data) > 0:
            can_avg = canada_data['avg'].values[0]
            rank = (country_stats['avg'] > can_avg).sum() + 1
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("🍁 Canada Avg Inflation", f"{can_avg:.2f}%")
            with col2:
                st.metric("🍁 Canada Rank", f"#{rank}", f"out of {len(country_stats)}")
        
    except Exception as e:
        st.error(f"Error: {str(e)}")

st.markdown("---")

# INSIGHT 3: Temporal Trends
st.markdown('<div class="insight-box">', unsafe_allow_html=True)
st.header("📅 Insight 3: Inflation Evolution (2020-2024)")
st.markdown("""
Tracks inflation trends over time, identifying peaks and patterns.
""")
st.markdown('</div>', unsafe_allow_html=True)

try:
    cpi_ts = cpi_df.copy()
    cpi_ts = cpi_ts.sort_values(cpi_time)
    
    key_countries = ['CAN', 'USA', 'GBR', 'DEU', 'FRA', 'JPN']
    available = [c for c in key_countries if c in cpi_ts[cpi_country].unique()]
    
    if not available:
        available = cpi_ts[cpi_country].value_counts().head(6).index.tolist()
    
    cpi_plot = cpi_ts[cpi_ts[cpi_country].isin(available)]
    
    fig = px.line(cpi_plot, x=cpi_time, y=cpi_value, color=cpi_country,
                  title='Monthly Inflation Trends',
                  labels={cpi_value: 'Inflation Rate (%)', cpi_time: 'Date'},
                  color_discrete_sequence=px.colors.qualitative.Bold)
    
    for trace in fig.data:
        if 'CAN' in trace.name:
            trace.update(line=dict(width=4))
    
    fig.update_layout(height=500, hovermode='x unified')
    st.plotly_chart(fig, use_container_width=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        canada_avg = cpi_plot[cpi_plot[cpi_country].str.contains('CAN', case=False, na=False)][cpi_value].mean()
        st.metric("🍁 Canada Avg", f"{canada_avg:.2f}%")
    
    with col2:
        overall_avg = cpi_plot[cpi_value].mean()
        st.metric("OECD Avg", f"{overall_avg:.2f}%")
    
    with col3:
        peak = cpi_plot[cpi_value].max()
        st.metric("Peak Inflation", f"{peak:.2f}%")
    
    with col4:
        recent = cpi_plot[cpi_plot[cpi_time] >= '2024-01-01'][cpi_value].mean()
        st.metric("2024 Avg", f"{recent:.2f}%")
    
except Exception as e:
    st.error(f"Error: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #7f8c8d; padding: 20px;'>
<p><strong>Inflation and Household Consumption Analysis Dashboard</strong></p>
<p>Data Source: OECD | Analysis Period: 2020-2024 | Created by Aishu for ALY6080</p>
</div>
""", unsafe_allow_html=True)
