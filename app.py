"""
Inflation and Household Consumption in Canada: A Comparative OECD Analysis Dashboard
Author: Aishwayra
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
    # GitHub Release direct download links
    CPI_FILE = "https://github.com/aishwaryashan20/dashboard_creation/releases/download/v1.0/Monthly.Consumer.Price.Indices.CPI.HICP.csv"
    CONSUMPTION_FILE = "https://github.com/aishwaryashan20/dashboard_creation/releases/download/v1.0/Annual.Household.Final.Consumption.Expenditure.COICOP.csv"
else:
    # Local file paths (for local testing)
    CPI_FILE = "Monthly Consumer Price Indices (CPI, HICP).csv"
    CONSUMPTION_FILE = "Annual Household Final Consumption Expenditure (COICOP).csv"

@st.cache_data
def load_and_process_cpi(file_path):
    """Load and automatically process CPI dataset"""
    try:
        # For Google Drive URLs, use different loading approach
        if file_path.startswith('http'):
            st.sidebar.info("📥 Downloading CPI data from Google Drive...")
            df = pd.read_csv(file_path, low_memory=False, on_bad_lines='skip', encoding='utf-8')
        else:
            df = pd.read_csv(file_path, low_memory=False, on_bad_lines='skip')
        
        st.sidebar.info(f"📂 CPI file loaded: {len(df):,} total records")
        
        # Show first few rows for debugging
        with st.sidebar.expander("🔍 Debug: View CPI Columns"):
            st.write("**Available columns:**")
            st.write(df.columns.tolist())
            st.write("**Sample data:**")
            st.dataframe(df.head(3))
        
        # Auto-detect columns - try exact matches first
        country_col = None
        time_col = None
        value_col = None
        
        # Priority matches for country
        for col in ['REF_AREA', 'LOCATION', 'Country', 'COUNTRY']:
            if col in df.columns:
                country_col = col
                break
        
        # Priority matches for time
        for col in ['TIME_PERIOD', 'Time', 'TIME', 'Date', 'PERIOD']:
            if col in df.columns:
                time_col = col
                break
        
        # Priority matches for value
        for col in ['OBS_VALUE', 'Value', 'VALUE', 'Observation']:
            if col in df.columns:
                value_col = col
                break
        
        # If still not found, search by pattern
        if country_col is None:
            for col in df.columns:
                if any(x in col.upper() for x in ['AREA', 'COUNTRY', 'LOCATION']):
                    country_col = col
                    break
        
        if time_col is None:
            for col in df.columns:
                if any(x in col.upper() for x in ['TIME', 'DATE', 'PERIOD']):
                    time_col = col
                    break
        
        if value_col is None:
            for col in df.columns:
                if any(x in col.upper() for x in ['VALUE', 'OBS']):
                    value_col = col
                    break
        
        st.sidebar.success(f"✅ Detected columns:\n- Country: {country_col}\n- Time: {time_col}\n- Value: {value_col}")
        
        # Convert value to numeric first
        df[value_col] = pd.to_numeric(df[value_col], errors='coerce')
        
        # Try multiple date formats
        df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
        
        # Check how many dates were parsed
        valid_dates = df[time_col].notna().sum()
        st.sidebar.info(f"📅 Valid dates parsed: {valid_dates:,}")
        
        # Filter 2020-2024
        df_filtered = df[
            (df[time_col].notna()) & 
            (df[time_col].dt.year >= 2020) & 
            (df[time_col].dt.year <= 2024)
        ].copy()
        
        st.sidebar.success(f"✅ CPI Data: {len(df_filtered):,} records (2020-2024)")
        
        return df_filtered, country_col, time_col, value_col
        
    except Exception as e:
        st.sidebar.error(f"❌ Error loading CPI data: {str(e)}")
        import traceback
        st.sidebar.code(traceback.format_exc())
        return None, None, None, None

@st.cache_data
def load_and_process_consumption(file_path):
    """Load and automatically process Consumption dataset"""
    try:
        # For Google Drive URLs, use different loading approach
        if file_path.startswith('http'):
            st.sidebar.info("📥 Downloading Consumption data from Google Drive...")
            df = pd.read_csv(file_path, low_memory=False, on_bad_lines='skip', encoding='utf-8')
        else:
            df = pd.read_csv(file_path, low_memory=False, on_bad_lines='skip')
        
        st.sidebar.info(f"📂 Consumption file loaded: {len(df):,} total records")
        
        # Show first few rows for debugging
        with st.sidebar.expander("🔍 Debug: View Consumption Columns"):
            st.write("**Available columns:**")
            st.write(df.columns.tolist())
            st.write("**Sample data:**")
            st.dataframe(df.head(3))
        
        # Auto-detect columns - exact matches first
        country_col = None
        time_col = None
        value_col = None
        
        # Priority matches
        for col in ['REF_AREA', 'LOCATION', 'Country', 'COUNTRY']:
            if col in df.columns:
                country_col = col
                break
        
        for col in ['TIME_PERIOD', 'Time', 'TIME', 'Date', 'PERIOD']:
            if col in df.columns:
                time_col = col
                break
        
        for col in ['OBS_VALUE', 'Value', 'VALUE', 'Observation']:
            if col in df.columns:
                value_col = col
                break
        
        # Pattern search if not found
        if country_col is None:
            for col in df.columns:
                if any(x in col.upper() for x in ['AREA', 'COUNTRY', 'LOCATION']):
                    country_col = col
                    break
        
        if time_col is None:
            for col in df.columns:
                if any(x in col.upper() for x in ['TIME', 'DATE', 'PERIOD']):
                    time_col = col
                    break
        
        if value_col is None:
            for col in df.columns:
                if any(x in col.upper() for x in ['VALUE', 'OBS']):
                    value_col = col
                    break
        
        st.sidebar.success(f"✅ Detected columns:\n- Country: {country_col}\n- Time: {time_col}\n- Value: {value_col}")
        
        # Convert types
        df[value_col] = pd.to_numeric(df[value_col], errors='coerce')
        df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
        
        # Check parsed dates
        valid_dates = df[time_col].notna().sum()
        st.sidebar.info(f"📅 Valid dates parsed: {valid_dates:,}")
        
        # Filter 2020-2024
        df_filtered = df[
            (df[time_col].notna()) & 
            (df[time_col].dt.year >= 2020) & 
            (df[time_col].dt.year <= 2024)
        ].copy()
        
        st.sidebar.success(f"✅ Consumption Data: {len(df_filtered):,} records (2020-2024)")
        
        return df_filtered, country_col, time_col, value_col
        
    except Exception as e:
        st.sidebar.error(f"❌ Error loading Consumption data: {str(e)}")
        import traceback
        st.sidebar.code(traceback.format_exc())
        return None, None, None, None

# Load datasets
with st.spinner("🔄 Loading and processing OECD datasets..."):
    cpi_df, cpi_country, cpi_time, cpi_value = load_and_process_cpi(CPI_FILE)
    cons_df, cons_country, cons_time, cons_value = load_and_process_consumption(CONSUMPTION_FILE)

# Check if data loaded
if cpi_df is None or cons_df is None:
    st.error("⚠️ Could not load datasets. Please ensure both CSV files are in the same folder as app.py")
    st.info(f"Looking for:\n- {CPI_FILE}\n- {CONSUMPTION_FILE}")
    st.stop()

# Sidebar info
with st.sidebar:
    st.header("📋 Automatic Analysis")
    st.success("✅ Datasets loaded and processed")
    st.markdown("""
    **Three Key Insights:**
    1. 🎯 Country Clustering
    2. 📈 Inflation-Consumption Correlation
    3. 📅 Temporal Trends (2020-2024)
    
    **Focus:** Canada vs OECD Countries
    """)
    st.markdown("---")
    st.info(f"""
    **Detected Columns:**
    
    CPI Data:
    - Country: `{cpi_country}`
    - Time: `{cpi_time}`
    - Value: `{cpi_value}`
    
    Consumption Data:
    - Country: `{cons_country}`
    - Time: `{cons_time}`
    - Value: `{cons_value}`
    """)

# Dataset Overview
st.header("📊 Dataset Overview (2020-2024)")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("CPI Records", f"{len(cpi_df):,}")
with col2:
    st.metric("Consumption Records", f"{len(cons_df):,}")
with col3:
    countries_cpi = cpi_df[cpi_country].nunique()
    st.metric("Countries (CPI)", countries_cpi)
with col4:
    countries_cons = cons_df[cons_country].nunique()
    st.metric("Countries (Consumption)", countries_cons)

st.markdown("---")

# INSIGHT 1: Clustering Analysis
st.markdown('<div class="insight-box">', unsafe_allow_html=True)
st.header("🎯 Insight 1: Country Clustering by Inflation Patterns")
st.markdown("""
**Purpose:** Groups OECD countries based on their inflation behavior since 2020.
Countries with similar inflation patterns (average level, volatility, peak) are clustered together.

**Key Finding:** Identifies which countries faced similar inflationary pressures and where Canada fits among OECD nations.
""")
st.markdown('</div>', unsafe_allow_html=True)

try:
    # Aggregate inflation metrics by country
    cluster_data = cpi_df.groupby(cpi_country)[cpi_value].agg([
        ('avg_inflation', 'mean'),
        ('volatility', 'std'),
        ('peak_inflation', 'max'),
        ('min_inflation', 'min')
    ]).dropna()
    
    if len(cluster_data) >= 3:
        # Standardize for clustering
        scaler = StandardScaler()
        scaled = scaler.fit_transform(cluster_data)
        
        # K-means clustering
        n_clusters = min(4, len(cluster_data))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_data['Cluster'] = kmeans.fit_predict(scaled)
        cluster_data['Group'] = ['Group ' + str(i+1) for i in cluster_data['Cluster']]
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Cluster distribution
            cluster_counts = cluster_data['Group'].value_counts().reset_index()
            cluster_counts.columns = ['Group', 'Count']
            
            fig1 = px.bar(
                cluster_counts,
                x='Group', y='Count',
                title='Distribution of Countries Across Inflation Clusters',
                color='Group',
                color_discrete_sequence=px.colors.qualitative.Set2,
                text='Count'
            )
            fig1.update_traces(textposition='outside')
            fig1.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            # Scatter plot: Average vs Volatility
            fig2 = px.scatter(
                cluster_data.reset_index(),
                x='avg_inflation', 
                y='volatility',
                color='Group',
                size='peak_inflation',
                hover_data=[cpi_country, 'avg_inflation', 'volatility', 'peak_inflation'],
                title='Inflation Patterns: Average Rate vs Volatility',
                labels={
                    'avg_inflation': 'Average Inflation Rate (%)',
                    'volatility': 'Volatility (Std Dev)',
                    'peak_inflation': 'Peak Inflation'
                },
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            fig2.update_layout(height=400)
            st.plotly_chart(fig2, use_container_width=True)
        
        # Cluster details
        st.subheader("📋 Detailed Cluster Analysis")
        
        for i in range(n_clusters):
            cluster_countries = cluster_data[cluster_data['Cluster'] == i]
            countries_list = cluster_countries.index.tolist()
            
            avg_inf = cluster_countries['avg_inflation'].mean()
            avg_vol = cluster_countries['volatility'].mean()
            avg_peak = cluster_countries['peak_inflation'].mean()
            
            # Check if Canada is in this cluster
            is_canada = any('CAN' in str(c) or 'Canada' in str(c) for c in countries_list)
            
            with st.expander(f"**Group {i+1}** ({'🍁 INCLUDES CANADA' if is_canada else f'{len(countries_list)} countries'})", expanded=is_canada):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Avg Inflation", f"{avg_inf:.2f}%")
                with col2:
                    st.metric("Avg Volatility", f"{avg_vol:.2f}%")
                with col3:
                    st.metric("Avg Peak", f"{avg_peak:.2f}%")
                
                st.markdown("**Countries in this group:**")
                st.write(", ".join(countries_list))
                
                if is_canada:
                    st.success("🍁 **Canada's Cluster:** This group represents countries with similar inflation experiences to Canada during 2020-2024.")
    else:
        st.warning("Not enough countries for clustering analysis")
        
except Exception as e:
    st.error(f"Error in clustering: {str(e)}")

st.markdown("---")

# INSIGHT 2: Correlation Analysis
st.markdown('<div class="insight-box">', unsafe_allow_html=True)
st.header("📈 Insight 2: Inflation-Consumption Correlation")
st.markdown("""
**Purpose:** Examines the relationship between inflation rates and household consumption expenditure.

**Key Finding:** Shows whether higher inflation leads to reduced consumption (negative correlation) or if 
households maintain spending despite price increases.
""")
st.markdown('</div>', unsafe_allow_html=True)

try:
    # Prepare yearly aggregates for both datasets
    cpi_yearly = cpi_df.copy()
    cpi_yearly['Year'] = cpi_yearly[cpi_time].dt.year
    cpi_agg = cpi_yearly.groupby([cpi_country, 'Year'])[cpi_value].mean().reset_index()
    cpi_agg.columns = ['Country', 'Year', 'AvgInflation']
    
    cons_yearly = cons_df.copy()
    cons_yearly['Year'] = cons_yearly[cons_time].dt.year
    cons_agg = cons_yearly.groupby([cons_country, 'Year'])[cons_value].mean().reset_index()
    cons_agg.columns = ['Country', 'Year', 'AvgConsumption']
    
    # Show what we have before merging
    st.info(f"📊 CPI Data: {len(cpi_agg)} country-year combinations | Consumption Data: {len(cons_agg)} country-year combinations")
    
    with st.expander("🔍 View Countries in Each Dataset"):
        col1, col2 = st.columns(2)
        with col1:
            st.write("**CPI Countries:**")
            st.write(sorted(cpi_agg['Country'].unique()[:20]))
        with col2:
            st.write("**Consumption Countries:**")
            st.write(sorted(cons_agg['Country'].unique()[:20]))
    
    # Try fuzzy matching for countries (CAN vs Canada, USA vs United States, etc.)
    # Create a mapping of common country code variations
    country_mapping = {
        'CAN': 'Canada', 'USA': 'United States', 'GBR': 'United Kingdom',
        'DEU': 'Germany', 'FRA': 'France', 'ITA': 'Italy', 'JPN': 'Japan',
        'AUS': 'Australia', 'ESP': 'Spain', 'NLD': 'Netherlands', 'BEL': 'Belgium',
        'AUT': 'Austria', 'SWE': 'Sweden', 'NOR': 'Norway', 'DNK': 'Denmark',
        'FIN': 'Finland', 'PRT': 'Portugal', 'GRC': 'Greece', 'IRL': 'Ireland',
        'NZL': 'New Zealand', 'CHE': 'Switzerland', 'POL': 'Poland', 'CZE': 'Czech Republic',
        'HUN': 'Hungary', 'KOR': 'Korea', 'MEX': 'Mexico', 'TUR': 'Turkey',
        'CHL': 'Chile', 'ISL': 'Iceland', 'ISR': 'Israel', 'SVN': 'Slovenia',
        'SVK': 'Slovak Republic', 'EST': 'Estonia', 'LVA': 'Latvia', 'LTU': 'Lithuania'
    }
    
    # Reverse mapping too
    reverse_mapping = {v: k for k, v in country_mapping.items()}
    
    # Standardize country names in both datasets
    def standardize_country(country):
        if country in country_mapping:
            return country_mapping[country]
        elif country in reverse_mapping:
            return reverse_mapping[country]
        return country
    
    cpi_agg['Country_Std'] = cpi_agg['Country'].apply(standardize_country)
    cons_agg['Country_Std'] = cons_agg['Country'].apply(standardize_country)
    
    # Merge on standardized country names
    merged = pd.merge(
        cpi_agg, cons_agg,
        left_on=['Country_Std', 'Year'],
        right_on=['Country_Std', 'Year'],
        how='inner',
        suffixes=('_cpi', '_cons')
    )
    
    st.success(f"✅ Successfully merged: {len(merged)} data points across {merged['Country_Std'].nunique()} countries")
    
    if len(merged) > 5:
        # Remove outliers for better visualization
        merged_clean = merged[
            (merged['AvgInflation'] < merged['AvgInflation'].quantile(0.95)) &
            (merged['AvgConsumption'] < merged['AvgConsumption'].quantile(0.95))
        ]
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Scatter with trendline
            fig3 = px.scatter(
                merged_clean,
                x='AvgInflation',
                y='AvgConsumption',
                color='Country',
                title='Inflation Rate vs Household Consumption',
                labels={
                    'AvgInflation': 'Average Inflation Rate (%)',
                    'AvgConsumption': 'Average Consumption Expenditure'
                },
                trendline="ols",
                trendline_scope="overall"
            )
            fig3.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig3, use_container_width=True)
        
        with col2:
            # Box plot by year
            fig4 = px.box(
                merged,
                x='Year',
                y='AvgInflation',
                title='Inflation Distribution by Year',
                labels={'AvgInflation': 'Inflation Rate (%)'},
                color='Year',
                color_discrete_sequence=px.colors.sequential.Blues
            )
            fig4.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig4, use_container_width=True)
        
        # Calculate correlation
        correlation = merged['AvgInflation'].corr(merged['AvgConsumption'])
        
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
            direction = "Positive" if correlation > 0 else "Negative"
            st.metric("Direction", direction)
        
        # Interpretation
        if correlation > 0.5:
            st.info("📊 **Strong Positive Correlation:** Higher inflation is associated with higher consumption expenditure, possibly due to nominal spending increases or households maintaining consumption despite rising prices.")
        elif correlation > 0.3:
            st.info("📊 **Moderate Positive Correlation:** Some relationship exists between inflation and consumption, but other factors also play significant roles.")
        elif correlation < -0.3:
            st.info("📊 **Negative Correlation:** Higher inflation tends to reduce real household consumption, indicating price sensitivity.")
        else:
            st.info("📊 **Weak Correlation:** Inflation and consumption show limited direct relationship, suggesting consumption is driven by other factors.")
        
        # Canada-specific analysis
        canada_data = merged[merged['Country'].str.contains('CAN|Canada', case=False, na=False)]
        if len(canada_data) > 0:
            st.subheader("🍁 Canada-Specific Analysis")
            fig5 = px.scatter(
                canada_data,
                x='AvgInflation',
                y='AvgConsumption',
                size='Year',
                title='Canada: Inflation vs Consumption Over Time',
                labels={
                    'AvgInflation': 'Inflation Rate (%)',
                    'AvgConsumption': 'Consumption Expenditure'
                },
                text='Year'
            )
            fig5.update_traces(textposition='top center')
            st.plotly_chart(fig5, use_container_width=True)
    else:
        st.warning("Insufficient overlapping data for correlation analysis")
        
except Exception as e:
    st.error(f"Error in correlation: {str(e)}")

st.markdown("---")

# INSIGHT 3: Temporal Trends
st.markdown('<div class="insight-box">', unsafe_allow_html=True)
st.header("📅 Insight 3: Temporal Evolution of Inflation (2020-2024)")
st.markdown("""
**Purpose:** Tracks inflation trends over time, identifying peaks, recovery periods, and current status.

**Key Finding:** Reveals when inflation peaked post-pandemic, how Canada's trajectory compares to other OECD nations,
and whether inflation is stabilizing.
""")
st.markdown('</div>', unsafe_allow_html=True)

try:
    # Prepare time series
    cpi_ts = cpi_df.copy()
    cpi_ts = cpi_ts.sort_values(cpi_time)
    
    # Identify key countries
    key_countries = ['CAN', 'USA', 'GBR', 'DEU', 'FRA', 'JPN', 'ITA', 'AUS', 'ESP', 'NLD']
    available = [c for c in key_countries if c in cpi_ts[cpi_country].unique()]
    
    if not available:
        # Use top countries by data
        available = cpi_ts[cpi_country].value_counts().head(8).index.tolist()
    
    cpi_ts_plot = cpi_ts[cpi_ts[cpi_country].isin(available)]
    
    # Monthly time series
    fig6 = px.line(
        cpi_ts_plot,
        x=cpi_time,
        y=cpi_value,
        color=cpi_country,
        title='Monthly Inflation Trends: Canada vs Key OECD Countries',
        labels={
            cpi_value: 'Inflation Rate (%)',
            cpi_time: 'Date',
            cpi_country: 'Country'
        },
        color_discrete_sequence=px.colors.qualitative.Bold
    )
    
    # Highlight Canada
    for trace in fig6.data:
        if 'CAN' in trace.name:
            trace.update(line=dict(width=4, dash='solid'))
        else:
            trace.update(line=dict(width=2))
    
    fig6.update_layout(
        height=500,
        hovermode='x unified',
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02)
    )
    st.plotly_chart(fig6, use_container_width=True)
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        canada_avg = cpi_ts[cpi_ts[cpi_country].str.contains('CAN', case=False, na=False)][cpi_value].mean()
        st.metric("🍁 Canada Avg", f"{canada_avg:.2f}%")
    
    with col2:
        overall_avg = cpi_ts_plot[cpi_value].mean()
        st.metric("OECD Avg", f"{overall_avg:.2f}%")
    
    with col3:
        peak_val = cpi_ts_plot[cpi_value].max()
        peak_date = cpi_ts_plot.loc[cpi_ts_plot[cpi_value].idxmax(), cpi_time]
        st.metric("Peak Inflation", f"{peak_val:.2f}%", f"{peak_date.strftime('%b %Y')}")
    
    with col4:
        recent_avg = cpi_ts_plot[cpi_ts_plot[cpi_time] >= '2024-01-01'][cpi_value].mean()
        st.metric("2024 Avg", f"{recent_avg:.2f}%")
    
    # Year-over-year comparison
    st.subheader("📊 Annual Average Inflation Comparison")
    
    yearly_data = cpi_ts_plot.copy()
    yearly_data['Year'] = yearly_data[cpi_time].dt.year
    yearly_avg = yearly_data.groupby(['Year', cpi_country])[cpi_value].mean().reset_index()
    
    fig7 = px.bar(
        yearly_avg,
        x='Year',
        y=cpi_value,
        color=cpi_country,
        title='Average Annual Inflation by Country',
        labels={cpi_value: 'Average Inflation Rate (%)'},
        barmode='group',
        color_discrete_sequence=px.colors.qualitative.Bold
    )
    fig7.update_layout(height=400)
    st.plotly_chart(fig7, use_container_width=True)
    
    # Trend analysis
    st.subheader("📈 Inflation Trend Analysis")
    
    canada_trend = cpi_ts[cpi_ts[cpi_country].str.contains('CAN', case=False, na=False)].copy()
    if len(canada_trend) > 0:
        # Get 2022 vs 2024
        inf_2022 = canada_trend[canada_trend[cpi_time].dt.year == 2022][cpi_value].mean()
        inf_2024 = canada_trend[canada_trend[cpi_time].dt.year == 2024][cpi_value].mean()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Canada 2022 Avg", f"{inf_2022:.2f}%")
        with col2:
            change = inf_2024 - inf_2022
            st.metric("Canada 2024 Avg", f"{inf_2024:.2f}%", f"{change:+.2f}pp")
        
        if change < -1:
            st.success("✅ **Positive Trend:** Canada's inflation has decreased significantly from 2022 peak, showing economic stabilization.")
        elif change > 1:
            st.warning("⚠️ **Rising Trend:** Inflation continues to increase, requiring continued policy attention.")
        else:
            st.info("📊 **Stable Trend:** Inflation remains relatively stable compared to peak levels.")
    
except Exception as e:
    st.error(f"Error in temporal analysis: {str(e)}")

# Summary Section
st.markdown("---")
st.header("🎯 Key Takeaways")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    ### Clustering
    - Countries grouped by inflation patterns
    - Canada's position among OECD peers
    - Similar economic pressures identified
    """)

with col2:
    st.markdown("""
    ### Correlation
    - Inflation-consumption relationship
    - Consumer behavior under price pressure
    - Economic resilience indicators
    """)

with col3:
    st.markdown("""
    ### Trends
    - Post-pandemic inflation trajectory
    - Peak identification and timing
    - Current stabilization status
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #7f8c8d; padding: 20px;'>
<p><strong>Inflation and Household Consumption Analysis Dashboard</strong></p>
<p>Data Source: OECD | Analysis Period: 2020-2024 | Created by Aishu for ALY6080</p>
<p><em>Automatic analysis of inflation patterns and household consumption across OECD countries with focus on Canada.</em></p>
</div>
""", unsafe_allow_html=True)
