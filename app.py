"""
Inflation and Household Consumption in Canada: A Comparative OECD Analysis Dashboard
Author: Aishwarya
Course: ALY6110
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

# INSIGHT 2: Correlation Analysis on Overlapping Years
st.markdown('<div class="insight-box">', unsafe_allow_html=True)
st.header("📈 Insight 2: Inflation-Consumption Correlation")
st.markdown("""
**Purpose:** Examines the relationship between inflation rates and household consumption expenditure
using historically overlapping data periods.

**Key Finding:** Shows whether higher inflation leads to reduced consumption (negative correlation) or if 
households maintain spending despite price increases.
""")
st.markdown('</div>', unsafe_allow_html=True)

try:
    if cons_df is None or len(cons_df) == 0:
        st.warning("⚠️ Loading full consumption dataset to find overlapping years...")
        
        # Reload consumption without 2020-2024 filter to get all available years
        if USE_URLS:
            cons_full = pd.read_csv(CONSUMPTION_FILE, low_memory=False, on_bad_lines='skip')
        else:
            cons_full = pd.read_csv(CONSUMPTION_FILE, low_memory=False, on_bad_lines='skip')
        
        # Parse dates
        cons_full[cons_time] = pd.to_datetime(cons_full[cons_time], errors='coerce')
        cons_full[cons_value] = pd.to_numeric(cons_full[cons_value], errors='coerce')
        cons_full = cons_full[cons_full[cons_time].notna()].copy()
    else:
        # Reload to get all years, not just 2020-2024
        if USE_URLS:
            cons_full = pd.read_csv(CONSUMPTION_FILE, low_memory=False, on_bad_lines='skip')
        else:
            cons_full = pd.read_csv(CONSUMPTION_FILE, low_memory=False, on_bad_lines='skip')
        
        cons_full[cons_time] = pd.to_datetime(cons_full[cons_time], errors='coerce')
        cons_full[cons_value] = pd.to_numeric(cons_full[cons_value], errors='coerce')
        cons_full = cons_full[cons_full[cons_time].notna()].copy()
    
    # Reload CPI without filter too
    if USE_URLS:
        cpi_full = pd.read_csv(CPI_FILE, low_memory=False, on_bad_lines='skip')
    else:
        cpi_full = pd.read_csv(CPI_FILE, low_memory=False, on_bad_lines='skip')
    
    cpi_full[cpi_time] = pd.to_datetime(cpi_full[cpi_time], errors='coerce')
    cpi_full[cpi_value] = pd.to_numeric(cpi_full[cpi_value], errors='coerce')
    cpi_full = cpi_full[cpi_full[cpi_time].notna()].copy()
    
    # Find overlapping years
    cpi_full['Year'] = cpi_full[cpi_time].dt.year
    cons_full['Year'] = cons_full[cons_time].dt.year
    
    cpi_years = set(cpi_full['Year'].unique())
    cons_years = set(cons_full['Year'].unique())
    overlap_years = sorted(cpi_years & cons_years)
    
    st.info(f"📊 **Found {len(overlap_years)} overlapping years:** {min(overlap_years)} to {max(overlap_years)}")
    
    if len(overlap_years) > 5:
        # Filter to overlapping years
        cpi_overlap = cpi_full[cpi_full['Year'].isin(overlap_years)].copy()
        cons_overlap = cons_full[cons_full['Year'].isin(overlap_years)].copy()
        
        # Aggregate by country and year
        cpi_agg = cpi_overlap.groupby([cpi_country, 'Year'])[cpi_value].mean().reset_index()
        cpi_agg.columns = ['Country', 'Year', 'AvgInflation']
        
        cons_agg = cons_overlap.groupby([cons_country, 'Year'])[cons_value].mean().reset_index()
        cons_agg.columns = ['Country', 'Year', 'AvgConsumption']
        
        # Country name mapping
        country_mapping = {
            'CAN': 'Canada', 'USA': 'United States', 'GBR': 'United Kingdom',
            'DEU': 'Germany', 'FRA': 'France', 'ITA': 'Italy', 'JPN': 'Japan',
            'AUS': 'Australia', 'ESP': 'Spain', 'NLD': 'Netherlands', 'BEL': 'Belgium',
            'AUT': 'Austria', 'SWE': 'Sweden', 'NOR': 'Norway', 'DNK': 'Denmark',
            'FIN': 'Finland', 'PRT': 'Portugal', 'GRC': 'Greece', 'IRL': 'Ireland',
            'NZL': 'New Zealand', 'CHE': 'Switzerland', 'POL': 'Poland', 'CZE': 'Czech Republic'
        }
        
        reverse_mapping = {v: k for k, v in country_mapping.items()}
        
        def standardize_country(country):
            if country in country_mapping:
                return country_mapping[country]
            elif country in reverse_mapping:
                return reverse_mapping[country]
            return country
        
        cpi_agg['Country_Std'] = cpi_agg['Country'].apply(standardize_country)
        cons_agg['Country_Std'] = cons_agg['Country'].apply(standardize_country)
        
        # Merge datasets
        merged = pd.merge(
            cpi_agg, cons_agg,
            left_on=['Country_Std', 'Year'],
            right_on=['Country_Std', 'Year'],
            how='inner',
            suffixes=('_cpi', '_cons')
        )
        
        st.success(f"✅ Successfully merged: **{len(merged)} data points** across **{merged['Country_Std'].nunique()} countries** ({min(overlap_years)}-{max(overlap_years)})")
        
        if len(merged) > 10:
            # Remove outliers
            merged_clean = merged[
                (merged['AvgInflation'] < merged['AvgInflation'].quantile(0.98)) &
                (merged['AvgConsumption'] < merged['AvgConsumption'].quantile(0.98))
            ]
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Scatter plot with trendline
                fig3 = px.scatter(
                    merged_clean,
                    x='AvgInflation',
                    y='AvgConsumption',
                    color='Country_Std',
                    hover_data=['Year'],
                    title=f'Inflation vs Consumption ({min(overlap_years)}-{max(overlap_years)})',
                    labels={
                        'AvgInflation': 'Average Inflation Rate (%)',
                        'AvgConsumption': 'Average Consumption Expenditure'
                    },
                    trendline="ols",
                    trendline_scope="overall",
                    trendline_color_override="red"
                )
                fig3.update_layout(height=450, showlegend=False)
                st.plotly_chart(fig3, use_container_width=True)
            
            with col2:
                # Canada-specific time series if available
                canada_merged = merged[merged['Country_Std'].str.contains('Canad|CAN', case=False, na=False)]
                
                if len(canada_merged) > 0:
                    fig_can = go.Figure()
                    
                    fig_can.add_trace(go.Scatter(
                        x=canada_merged['Year'],
                        y=canada_merged['AvgInflation'],
                        name='Inflation Rate',
                        yaxis='y',
                        line=dict(color='#e74c3c', width=3)
                    ))
                    
                    # Normalize consumption to fit on same scale
                    cons_norm = (canada_merged['AvgConsumption'] / canada_merged['AvgConsumption'].max()) * canada_merged['AvgInflation'].max()
                    fig_can.add_trace(go.Scatter(
                        x=canada_merged['Year'],
                        y=cons_norm,
                        name='Consumption (normalized)',
                        yaxis='y',
                        line=dict(color='#3498db', width=3, dash='dash')
                    ))
                    
                    fig_can.update_layout(
                        title='🍁 Canada: Inflation vs Consumption Trends',
                        xaxis_title='Year',
                        yaxis_title='Value',
                        height=450,
                        hovermode='x unified',
                        legend=dict(x=0.01, y=0.99)
                    )
                    st.plotly_chart(fig_can, use_container_width=True)
                else:
                    # Year distribution if no Canada data
                    fig4 = px.box(
                        merged,
                        x='Year',
                        y='AvgInflation',
                        title='Inflation Distribution by Year',
                        color_discrete_sequence=['#3498db']
                    )
                    fig4.update_layout(height=450, showlegend=False)
                    st.plotly_chart(fig4, use_container_width=True)
            
            # Calculate correlation
            correlation = merged['AvgInflation'].corr(merged['AvgConsumption'])
            
            # Display metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Correlation Coefficient", f"{correlation:.3f}")
            
            with col2:
                if abs(correlation) > 0.5:
                    strength = "Strong"
                elif abs(correlation) > 0.3:
                    strength = "Moderate"
                else:
                    strength = "Weak"
                st.metric("Correlation Strength", strength)
            
            with col3:
                direction = "Positive ↗" if correlation > 0 else "Negative ↘"
                st.metric("Direction", direction)
            
            # Interpretation
            st.subheader("📊 Interpretation")
            
            if correlation > 0.5:
                st.success(f"""
                🔍 **Strong Positive Correlation** (r = {correlation:.3f})
                
                During {min(overlap_years)}-{max(overlap_years)}, higher inflation was associated with higher consumption expenditure. This suggests:
                - Households maintained nominal spending despite rising prices
                - Economic growth periods saw both inflation and increased consumption
                - Limited price sensitivity in overall consumption patterns
                """)
            elif correlation > 0.3:
                st.info(f"""
                🔍 **Moderate Positive Correlation** (r = {correlation:.3f})
                
                Some relationship exists between inflation and consumption during {min(overlap_years)}-{max(overlap_years)}:
                - Mixed household responses to price changes
                - Other economic factors (income, employment) also influence consumption
                - Country-specific variations in price sensitivity
                """)
            elif correlation < -0.3:
                st.warning(f"""
                🔍 **Negative Correlation** (r = {correlation:.3f})
                
                Higher inflation tended to reduce consumption during {min(overlap_years)}-{max(overlap_years)}:
                - Real purchasing power erosion affected spending
                - Households cut back on consumption when prices rose
                - Price sensitivity in consumer behavior
                """)
            else:
                st.info(f"""
                🔍 **Weak Correlation** (r = {correlation:.3f})
                
                Inflation and consumption showed limited direct relationship during {min(overlap_years)}-{max(overlap_years)}:
                - Consumption primarily driven by other factors (income, employment, confidence)
                - Diverse country-specific patterns
                - Complex economic dynamics beyond price changes
                """)
            
            # Country-level analysis
            st.subheader("🌍 Country-Level Correlation Analysis")
            
            country_corr = merged.groupby('Country_Std').apply(
                lambda x: x['AvgInflation'].corr(x['AvgConsumption']) if len(x) >= 3 else np.nan
            ).dropna().sort_values(ascending=False)
            
            if len(country_corr) > 0:
                fig5 = px.bar(
                    x=country_corr.index[:15],
                    y=country_corr.values[:15],
                    title='Top 15 Countries: Inflation-Consumption Correlation',
                    labels={'x': 'Country', 'y': 'Correlation Coefficient'},
                    color=country_corr.values[:15],
                    color_continuous_scale='RdBu_r',
                    color_continuous_midpoint=0
                )
                fig5.update_layout(height=400, xaxis_tickangle=-45)
                st.plotly_chart(fig5, use_container_width=True)
                
                # Highlight Canada
                canada_corr = country_corr[country_corr.index.str.contains('Canad|CAN', case=False, na=False)]
                if len(canada_corr) > 0:
                    can_rank = list(country_corr.index).index(canada_corr.index[0]) + 1
                    st.success(f"🍁 **Canada's Correlation:** {canada_corr.values[0]:.3f} - Ranks **#{can_rank}** out of {len(country_corr)} countries")
        
        else:
            st.warning(f"⚠️ Only {len(merged)} overlapping data points - need more for meaningful analysis")
    
    else:
        st.error("❌ Insufficient overlapping years between datasets for correlation analysis")

except Exception as e:
    st.error(f"Error in correlation analysis: {str(e)}")
    import traceback
    st.code(traceback.format_exc())

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
