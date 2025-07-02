import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Set page config
st.set_page_config(
    page_title="SheStock: ESG Stock Recommender", 
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sample real stock data with ESG and diversity metrics (now including Indian stocks)
@st.cache_data
def load_real_stock_data():
    """Load or create real stock dataset with ESG and diversity data"""
    stock_data = {
        'Company': [
            # US Stocks
            'Microsoft Corporation', 'Apple Inc.', 'Alphabet Inc.', 'Tesla Inc.', 'Johnson & Johnson',
            'Procter & Gamble', 'Unilever PLC', 'JPMorgan Chase', 'Bank of America', 'Pfizer Inc.',
            'Merck & Co.', 'Visa Inc.', 'Mastercard Inc.', 'Cisco Systems', 'Intel Corporation',
            'NVIDIA Corporation', 'Advanced Micro Devices', 'Salesforce Inc.', 'Adobe Inc.', 'Netflix Inc.',
            
            # Indian Stocks
            'Reliance Industries', 'Tata Consultancy Services', 'HDFC Bank', 'Infosys', 'ICICI Bank',
            'Hindustan Unilever', 'Bharti Airtel', 'ITC Limited', 'State Bank of India', 'Kotak Mahindra Bank',
            'Asian Paints', 'HCL Technologies', 'Wipro', 'Larsen & Toubro', 'Axis Bank',
            'Maruti Suzuki', 'Tata Motors', 'Sun Pharmaceutical', 'Bajaj Finance', 'Nestle India',
            'Tata Steel', 'Tech Mahindra', 'UltraTech Cement', 'Mahindra & Mahindra', 'Adani Ports',
            'Bajaj Finserv', 'Britannia Industries', 'Grasim Industries', 'JSW Steel', 'Titan Company'
        ],
        'Ticker': [
            # US Tickers
            'MSFT', 'AAPL', 'GOOGL', 'TSLA', 'JNJ',
            'PG', 'UL', 'JPM', 'BAC', 'PFE',
            'MRK', 'V', 'MA', 'CSCO', 'INTC',
            'NVDA', 'AMD', 'CRM', 'ADBE', 'NFLX',
            
            # Indian Tickers (with .NS for NSE)
            'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS',
            'HINDUNILVR.NS', 'BHARTIARTL.NS', 'ITC.NS', 'SBIN.NS', 'KOTAKBANK.NS',
            'ASIANPAINT.NS', 'HCLTECH.NS', 'WIPRO.NS', 'LT.NS', 'AXISBANK.NS',
            'MARUTI.NS', 'TATAMOTORS.NS', 'SUNPHARMA.NS', 'BAJFINANCE.NS', 'NESTLEIND.NS',
            'TATASTEEL.NS', 'TECHM.NS', 'ULTRACEMCO.NS', 'M&M.NS', 'ADANIPORTS.NS',
            'BAJAJFINSV.NS', 'BRITANNIA.NS', 'GRASIM.NS', 'JSWSTEEL.NS', 'TITAN.NS'
        ],
        'Country': [
            # US Stocks
            'USA', 'USA', 'USA', 'USA', 'USA',
            'USA', 'UK', 'USA', 'USA', 'USA',
            'USA', 'USA', 'USA', 'USA', 'USA',
            'USA', 'USA', 'USA', 'USA', 'USA',
            
            # Indian Stocks
            'India', 'India', 'India', 'India', 'India',
            'India', 'India', 'India', 'India', 'India',
            'India', 'India', 'India', 'India', 'India',
            'India', 'India', 'India', 'India', 'India',
            'India', 'India', 'India', 'India', 'India',
            'India', 'India', 'India', 'India', 'India'
        ],
        'Female_CEO': [
            # US Stocks
            0, 0, 0, 0, 0,
            1, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
            0, 1, 0, 0, 0,
            
            # Indian Stocks (currently only HUL has female CEO)
            0, 0, 0, 0, 0,
            1, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 0
        ],
        'Female_Board_Members': [
            # US Stocks
            4, 3, 2, 2, 4,
            5, 4, 3, 4, 3,
            3, 3, 2, 3, 2,
            1, 4, 3, 3, 4,
            
            # Indian Stocks
            3, 2, 2, 3, 2,
            4, 2, 1, 2, 1,
            2, 1, 2, 2, 1,
            2, 1, 1, 1, 1,
            1, 1, 1, 1, 1,
            1, 1, 1, 1, 1
        ],
        'ESG_Score': [
            # US Stocks
            0.89, 0.75, 0.72, 0.85, 0.88,
            0.92, 0.91, 0.68, 0.71, 0.82,
            0.85, 0.74, 0.73, 0.78, 0.69,
            0.76, 0.71, 0.87, 0.81, 0.77,
            
            # Indian Stocks (estimated ESG scores)
            0.82, 0.85, 0.78, 0.87, 0.75,
            0.88, 0.72, 0.83, 0.68, 0.76,
            0.81, 0.84, 0.79, 0.77, 0.74,
            0.75, 0.72, 0.80, 0.82, 0.86,
            0.71, 0.83, 0.70, 0.73, 0.69,
            0.81, 0.85, 0.78, 0.72, 0.84
        ],
        'Sector': [
            # US Stocks
            'IT', 'IT', 'IT', 'Energy', 'Healthcare',
            'FMCG', 'FMCG', 'Finance', 'Finance', 'Pharma',
            'Pharma', 'Finance', 'Finance', 'IT', 'IT',
            'IT', 'IT', 'IT', 'IT', 'IT',
            
            # Indian Stocks
            'Conglomerate', 'IT', 'Finance', 'IT', 'Finance',
            'FMCG', 'Telecom', 'FMCG', 'Finance', 'Finance',
            'FMCG', 'IT', 'IT', 'Construction', 'Finance',
            'Automobile', 'Automobile', 'Pharma', 'Finance', 'FMCG',
            'Metals', 'IT', 'Construction', 'Automobile', 'Infrastructure',
            'Finance', 'FMCG', 'Cement', 'Metals', 'Retail'
        ]
    }
    
    df = pd.DataFrame(stock_data)
    
    # Create sector dummy variables
    sector_dummies = pd.get_dummies(df['Sector'], prefix='Sector')
    
    # Ensure all expected sectors are present
    expected_sectors = ['Automobile', 'Cement', 'Conglomerate', 'Construction', 'Energy', 
                       'FMCG', 'Finance', 'Healthcare', 'Infrastructure', 'IT', 
                       'Insurance', 'Metals', 'Mining', 'Pharma', 'Retail', 'Telecom']
    
    for sector in expected_sectors:
        col_name = f'Sector_{sector}'
        if col_name not in sector_dummies.columns:
            sector_dummies[col_name] = 0
    
    # Combine with original data
    result_df = pd.concat([df, sector_dummies], axis=1)
    
    return result_df

# [Rest of your existing code remains the same, but update the filters to include Country]

# In the Real Stock Matches tab (tab2), update the filters to include Country:
with tab2:
    st.header("📊 Real Stock Matches")
    st.markdown("Find real companies that match your investment criteria:")
    
    # Filter options - add Country filter
    col1, col2, col3 = st.columns(3)
    
    with col1:
        min_esg = st.slider("Minimum ESG Score", 0.0, 1.0, 0.7, key="min_esg")
        female_ceo_filter = st.selectbox("Female CEO Required?", ["Any", "Yes", "No"], key="ceo_filter")
        country_filter = st.selectbox("Country", ["All", "USA", "India", "UK"], key="country_filter")
    
    with col2:
        min_female_board = st.slider("Min Female Board Members", 0, 6, 2, key="min_board")
        sector_filter = st.selectbox("Sector Filter", ["All"] + list(real_stocks_df['Sector'].unique()), key="sector_filter")
    
    with col3:
        sort_by = st.selectbox("Sort by", ["ESG Score", "Female Leadership", "Company Name"], key="sort_by")
        max_results = st.slider("Max Results", 5, 50, 20, key="max_results")
    
    # Apply filters - add Country filter
    filtered_stocks = real_stocks_df.copy()
    
    # ESG filter
    filtered_stocks = filtered_stocks[filtered_stocks['ESG_Score'] >= min_esg]
    
    # Female CEO filter
    if female_ceo_filter == "Yes":
        filtered_stocks = filtered_stocks[filtered_stocks['Female_CEO'] == 1]
    elif female_ceo_filter == "No":
        filtered_stocks = filtered_stocks[filtered_stocks['Female_CEO'] == 0]
    
    # Country filter
    if country_filter != "All":
        filtered_stocks = filtered_stocks[filtered_stocks['Country'] == country_filter]
    
    # [Rest of the tab2 code remains the same]

# [Rest of your existing code remains the same]
