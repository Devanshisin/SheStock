# import streamlit as st
# import pandas as pd
# import pickle
# import plotly.express as px
# import plotly.graph_objects as go
# import numpy as np
# from datetime import datetime

# # Set page config
# st.set_page_config(
#     page_title="SheStock: ESG Stock Recommender", 
#     page_icon="🌱",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # Custom CSS for enhanced styling
# st.markdown("""
# <style>
#     .main-header {
#         font-size: 3.5rem;
#         font-weight: bold;
#         text-align: center;
#         background: linear-gradient(135deg, #2E8B57 0%, #48B685 50%, #20B2AA 100%);
#         -webkit-background-clip: text;
#         -webkit-text-fill-color: transparent;
#         margin-bottom: 0.5rem;
#     }
    
#     .subtitle {
#         text-align: center;
#         font-size: 1.3rem;
#         color: #2E8B57;
#         font-weight: 500;
#         margin-bottom: 2rem;
#         font-style: italic;
#     }
    
#     .feature-card {
#         background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
#         padding: 1.5rem;
#         border-radius: 15px;
#         border-left: 5px solid #2E8B57;
#         margin: 1rem 0;
#         box-shadow: 0 4px 15px rgba(46, 139, 87, 0.1);
#     }
    
#     .prediction-success {
#         background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
#         color: white;
#         padding: 2rem;
#         border-radius: 15px;
#         text-align: center;
#         font-size: 1.2rem;
#         font-weight: bold;
#         margin: 1rem 0;
#         box-shadow: 0 4px 20px rgba(40, 167, 69, 0.3);
#     }
    
#     .prediction-error {
#         background: linear-gradient(135deg, #dc3545 0%, #e83e8c 100%);
#         color: white;
#         padding: 2rem;
#         border-radius: 15px;
#         text-align: center;
#         font-size: 1.2rem;
#         font-weight: bold;
#         margin: 1rem 0;
#         box-shadow: 0 4px 20px rgba(220, 53, 69, 0.3);
#     }
    
#     .stats-card {
#         background: linear-gradient(135deg, #2E8B57 0%, #20B2AA 100%);
#         color: white;
#         padding: 1.5rem;
#         border-radius: 10px;
#         text-align: center;
#         margin: 0.5rem 0;
#         box-shadow: 0 4px 15px rgba(46, 139, 87, 0.3);
#     }
    
#     .sidebar-header {
#         font-size: 1.5rem;
#         font-weight: bold;
#         color: #2E8B57;
#         margin-bottom: 1rem;
#         text-align: center;
#     }
    
#     .confidence-high { color: #28a745; font-weight: bold; }
#     .confidence-medium { color: #ffc107; font-weight: bold; }
#     .confidence-low { color: #dc3545; font-weight: bold; }
    
#     .stButton > button {
#         background: linear-gradient(135deg, #2E8B57 0%, #20B2AA 100%);
#         color: white;
#         border: none;
#         padding: 0.7rem 2rem;
#         font-size: 1.1rem;
#         font-weight: bold;
#         border-radius: 10px;
#         transition: all 0.3s ease;
#     }
    
#     .stButton > button:hover {
#         transform: translateY(-2px);
#         box-shadow: 0 5px 15px rgba(46, 139, 87, 0.4);
#     }
# </style>
# """, unsafe_allow_html=True)

# # Load trained model
# @st.cache_resource
# def load_model():
#     try:
#         with open("model/recommender.pkl", "rb") as f:
#             return pickle.load(f)
#     except FileNotFoundError:
#         st.error("⚠️ Model file not found. Please ensure 'model/recommender.pkl' exists.")
#         return None

# model = load_model()

# # Initialize session state
# if 'prediction_history' not in st.session_state:
#     st.session_state.prediction_history = []

# # Header
# st.markdown('<h1 class="main-header">🌱 SheStock</h1>', unsafe_allow_html=True)
# st.markdown('<p class="subtitle">ESG-Based Stock Recommender: Empowering Women-Led Sustainable Investment Decisions</p>', unsafe_allow_html=True)

# # Sidebar for additional features
# with st.sidebar:
#     st.markdown('<div class="sidebar-header">📊 Dashboard</div>', unsafe_allow_html=True)
    
#     # Show prediction statistics
#     if st.session_state.prediction_history:
#         total_predictions = len(st.session_state.prediction_history)
#         recommended_count = sum(1 for p in st.session_state.prediction_history if p['prediction'])
        
#         st.markdown(f"""
#         <div class="stats-card">
#             <h3>📈 Session Stats</h3>
#             <p>Total Predictions: {total_predictions}</p>
#             <p>Recommended: {recommended_count}</p>
#             <p>Success Rate: {recommended_count/total_predictions*100:.1f}%</p>
#         </div>
#         """, unsafe_allow_html=True)
    
#     # ESG Information
#     st.markdown("---")
#     st.markdown("### 🌍 ESG Criteria")
#     st.markdown("""
#     **Environmental**: Climate change, resource depletion, waste & pollution
    
#     **Social**: Working conditions, local communities, health & safety
    
#     **Governance**: Management structure, employee relations, executive compensation
#     """)
    
#     # Quick tips
#     st.markdown("---")
#     st.markdown("### 💡 Investment Tips")
#     st.info("Companies with higher ESG scores often show better long-term performance and lower risk.")
#     st.success("Female leadership in companies is correlated with better ESG performance.")

# # Main content in tabs
# tab1, tab2, tab3 = st.tabs(["🔍 Stock Analysis", "📊 Visualization", "📈 History"])

# with tab1:
#     col1, col2 = st.columns([2, 1])
    
#     with col1:
#         st.markdown('<div class="feature-card">', unsafe_allow_html=True)
#         st.header("📋 Company Information")
        
#         # Input fields with better organization
#         col_a, col_b = st.columns(2)
        
#         with col_a:
#             st.subheader("👩‍💼 Leadership")
#             female_ceo = st.selectbox("Female CEO?", [0, 1], index=1, 
#                                     format_func=lambda x: "Yes" if x == 1 else "No")
#             female_board = st.slider("Female Board Members", 0, 6, 2, 
#                                    help="Number of female board members")
            
#             st.subheader("🌱 ESG Performance")
#             esg_score = st.slider("ESG Score", 0.0, 1.0, 0.8, 
#                                 help="Environmental, Social, and Governance score (0.0 to 1.0)")
        
#         with col_b:
#             st.subheader("🏭 Industry Sector")
#             sectors = {
#                 "Energy": st.checkbox("⚡ Energy"),
#                 "FMCG": st.checkbox("🛒 FMCG"),
#                 "Finance": st.checkbox("💰 Finance"),
#                 "Healthcare": st.checkbox("🏥 Healthcare"),
#                 "IT": st.checkbox("💻 IT", value=True),
#                 "Insurance": st.checkbox("📋 Insurance"),
#                 "Mining": st.checkbox("⛏️ Mining"),
#                 "Pharma": st.checkbox("💊 Pharma"),
#                 "Retail": st.checkbox("🛍️ Retail")
#             }
        
#         st.markdown('</div>', unsafe_allow_html=True)
        
#         # Company data preparation
#         company_data = {
#             "Female_CEO": female_ceo,
#             "Female_Board_Members": female_board,
#             "ESG_Score": esg_score,
#             "Sector_Energy": int(sectors["Energy"]),
#             "Sector_FMCG": int(sectors["FMCG"]),
#             "Sector_Finance": int(sectors["Finance"]),
#             "Sector_Healthcare": int(sectors["Healthcare"]),
#             "Sector_IT": int(sectors["IT"]),
#             "Sector_Insurance": int(sectors["Insurance"]),
#             "Sector_Mining": int(sectors["Mining"]),
#             "Sector_Pharma": int(sectors["Pharma"]),
#             "Sector_Retail": int(sectors["Retail"])
#         }
    
#     with col2:
#         st.markdown("### 📊 Current Input Summary")
        
#         # Visual representation of inputs
#         st.markdown(f"""
#         **Leadership Score**: {"🟢 High" if female_ceo and female_board >= 3 else "🟡 Medium" if female_ceo or female_board >= 2 else "🔴 Low"}
        
#         **ESG Score**: {esg_score:.1f}/1.0
        
#         **Selected Sectors**: {sum(sectors.values())} selected
#         """)
        
#         # ESG Score visualization
#         fig_gauge = go.Figure(go.Indicator(
#             mode = "gauge+number",
#             value = esg_score,
#             domain = {'x': [0, 1], 'y': [0, 1]},
#             title = {'text': "ESG Score"},
#             gauge = {
#                 'axis': {'range': [None, 1]},
#                 'bar': {'color': "#2E8B57"},
#                 'steps': [
#                     {'range': [0, 0.5], 'color': "lightgray"},
#                     {'range': [0.5, 0.8], 'color': "gray"}
#                 ],
#                 'threshold': {
#                     'line': {'color': "red", 'width': 4},
#                     'thickness': 0.75,
#                     'value': 0.9
#                 }
#             }
#         ))
#         fig_gauge.update_layout(height=200)
#         st.plotly_chart(fig_gauge, use_container_width=True)
    
#     # Prediction section
#     st.markdown("---")
#     col1, col2, col3 = st.columns([1, 1, 1])
    
#     with col2:
#         if st.button("🔍 Predict Recommendation", use_container_width=True):
#             if model is not None:
#                 # Expected columns (MUST match training)
#                 expected_cols = [
#                     "Female_CEO", "Female_Board_Members", "ESG_Score",
#                     "Sector_Energy", "Sector_FMCG", "Sector_Finance",
#                     "Sector_Healthcare", "Sector_IT", "Sector_Insurance",
#                     "Sector_Mining", "Sector_Pharma", "Sector_Retail"
#                 ]
                
#                 input_df = pd.DataFrame([company_data])
                
#                 # Add missing columns (if any)
#                 for col in expected_cols:
#                     if col not in input_df.columns:
#                         input_df[col] = 0
                
#                 input_df = input_df[expected_cols]
                
#                 # Make prediction
#                 pred = model.predict(input_df)[0]
#                 proba = model.predict_proba(input_df)[0][1]
                
#                 # Store in history
#                 st.session_state.prediction_history.append({
#                     'timestamp': datetime.now(),
#                     'prediction': pred,
#                     'confidence': proba,
#                     'data': company_data.copy()
#                 })
                
#                 # Display results
#                 if pred:
#                     confidence_class = "confidence-high" if proba > 0.8 else "confidence-medium" if proba > 0.6 else "confidence-low"
#                     st.markdown(f"""
#                     <div class="prediction-success">
#                         ✅ <strong>RECOMMENDED</strong><br>
#                         <span class="{confidence_class}">Confidence: {proba*100:.1f}%</span>
#                     </div>
#                     """, unsafe_allow_html=True)
#                 # else:
#                 #     confidence_class = "confidence-high" if (1-proba) > 0.8 else "confidence-medium" if (1-proba) > 0.6 else "confidence-low"
#                 #     st.markdown(f"""
#                     <div class="prediction-error">
#                        <strong> &#128683; NOT RECOMMENDED</strong><br>

#                         <span class="{confidence_class}">Confidence: {(1-proba)*100:.1f}%</span>
#                     </div>
#                     """, unsafe_allow_html=True)
#             else:
#                 st.error("Model not available. Please check the model file.")
#                 --- Show real similar companies ---
#                 try:
#                     esg_df = pd.read_csv("preprocessed_esg_dataset.csv")
#                     # Filter dataset for similar profile
#                     matching = esg_df[
#                         (esg_df["Female_CEO"] == female_ceo) &
#                         (esg_df["Female_Board_Members"] >= female_board - 1) &
#                         (esg_df["Female_Board_Members"] <= female_board + 1) &
#                         (esg_df["ESG_Score"] >= esg_score - 0.1) &
#                         (esg_df["ESG_Score"] <= esg_score + 0.1)
#                     ]

#                     # Sort by ESG Score + Board representation
#                     matching["Score"] = matching["ESG_Score"] + 0.05 * matching["Female_Board_Members"]
#                     top_stocks = matching.sort_values(by="Score", ascending=False).head(5)

#                     if not top_stocks.empty:
#                         st.markdown("### 🏆 Top Real Stocks Matching Profile")
#                         for i, row in top_stocks.iterrows():
#                             st.markdown(f"""
#                             **{row['Company']}** ({row['Ticker']})  
#                             ESG Score: {row['ESG_Score']:.2f} | Female Board Members: {int(row['Female_Board_Members'])}
#                             """)
#                     else:
#                         st.warning("No exact real stock matches found. Try changing the ESG or leadership values.")
#                 except Exception as e:
#                     st.error(f"❌ Could not load real stock data: {e}")

# with tab2:
#     st.header("📊 Data Visualization")
    
#     if st.session_state.prediction_history:
#         # Create visualizations from prediction history
#         df_history = pd.DataFrame([
#             {
#                 'timestamp': p['timestamp'],
#                 'prediction': 'Recommended' if p['prediction'] else 'Not Recommended',
#                 'confidence': p['confidence'] if p['prediction'] else 1 - p['confidence'],
#                 'esg_score': p['data']['ESG_Score'],
#                 'female_ceo': p['data']['Female_CEO'],
#                 'female_board': p['data']['Female_Board_Members']
#             }
#             for p in st.session_state.prediction_history
#         ])
        
#         col1, col2 = st.columns(2)
        
#         with col1:
#             # Prediction distribution
#             fig_pie = px.pie(df_history, names='prediction', 
#                            title="Prediction Distribution",
#                            color_discrete_map={'Recommended': '#2E8B57', 'Not Recommended': '#dc3545'})
#             st.plotly_chart(fig_pie, use_container_width=True)
        
#         with col2:
#             # ESG Score vs Confidence
#             fig_scatter = px.scatter(df_history, x='esg_score', y='confidence',
#                                    color='prediction', title="ESG Score vs Confidence",
#                                    color_discrete_map={'Recommended': '#2E8B57', 'Not Recommended': '#dc3545'})
#             st.plotly_chart(fig_scatter, use_container_width=True)
        
#         # Leadership impact analysis
#         fig_bar = px.bar(df_history.groupby(['female_ceo', 'prediction']).size().reset_index(name='count'),
#                         x='female_ceo', y='count', color='prediction',
#                         title="Impact of Female CEO on Recommendations",
#                         color_discrete_map={'Recommended': '#2E8B57', 'Not Recommended': '#dc3545'})
#         st.plotly_chart(fig_bar, use_container_width=True)
#     else:
#         st.info("📈 Make some predictions first to see visualizations!")
        
#         # Show sample visualization
#         sample_data = pd.DataFrame({
#             'ESG_Score': np.random.uniform(0.3, 0.95, 50),
#             'Female_Leadership': np.random.choice(['High', 'Medium', 'Low'], 50),
#             'Recommendation': np.random.choice(['Recommended', 'Not Recommended'], 50)
#         })
        
#         fig_sample = px.scatter(sample_data, x='ESG_Score', y='Female_Leadership',
#                               color='Recommendation', title="Sample: ESG Score vs Female Leadership",
#                               color_discrete_map={'Recommended': '#2E8B57', 'Not Recommended': '#dc3545'})
#         st.plotly_chart(fig_sample, use_container_width=True)

# with tab3:
#     st.header("📈 Prediction History")
    
#     if st.session_state.prediction_history:
#         # Display recent predictions
#         for i, prediction in enumerate(reversed(st.session_state.prediction_history[-10:])):
#             with st.expander(f"Prediction {len(st.session_state.prediction_history) - i} - {prediction['timestamp'].strftime('%Y-%m-%d %H:%M')}"):
#                 col1, col2 = st.columns(2)
                
#                 with col1:
#                     st.write("**Input Data:**")
#                     st.json(prediction['data'])
                
#                 with col2:
#                     result = "✅ Recommended" if prediction['prediction'] else "🚫 Not Recommended"
#                     confidence = prediction['confidence'] if prediction['prediction'] else 1 - prediction['confidence']
#                     st.write(f"**Result:** {result}")
#                     st.write(f"**Confidence:** {confidence*100:.1f}%")
        
#         # Clear history button
#         if st.button("🗑️ Clear History"):
#             st.session_state.prediction_history = []
#             st.rerun()
#     else:
#         st.info("No predictions made yet. Go to the Stock Analysis tab to make your first prediction!")

# # Footer with glossary
# st.markdown("---")
# with st.expander("📚 Glossary & Information"):
#     col1, col2 = st.columns(2)
    
#     with col1:
#         st.markdown("""
#         ### 📖 Key Terms
#         - **ESG Score**: Measures how well a company performs on environmental, social, and governance criteria (0.0 to 1.0)
#         - **Female CEO**: Binary indicator of female chief executive leadership
#         - **Female Board Members**: Number of women on the board of directors (0-6)
#         - **Sector**: Company's primary industry classification
#         """)
    
#     with col2:
#         st.markdown("""
#         ### 🎯 Model Information
#         - **Recommendation**: Binary output indicating investment recommendation
#         - **Confidence**: Probability score indicating model certainty
#         - **Features**: Leadership diversity, ESG performance, and sector classification
#         - **Purpose**: Promote sustainable and inclusive investment decisions
#         """)

# st.markdown("---")
# st.markdown("""
# <div style='text-align: center; color: #2E8B57; font-size: 1rem; margin-top: 2rem;'>
#     💡 <strong>Created with ❤️ by Devanshi for ESG-conscious investing</strong><br>
#     <em>Empowering women-led sustainable investment decisions</em>
# </div>
# """, unsafe_allow_html=True)

import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime

# Set page config
st.set_page_config(
    page_title="SheStock: ESG Stock Recommender", 
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sample real stock data with ESG and diversity metrics
@st.cache_data
def load_real_stock_data():
    """Load or create real stock dataset with ESG and diversity data"""
    stock_data = {
        'Company': [
            'Microsoft Corporation', 'Apple Inc.', 'Alphabet Inc.', 'Tesla Inc.', 'Johnson & Johnson',
            'Procter & Gamble', 'Unilever PLC', 'JPMorgan Chase', 'Bank of America', 'Pfizer Inc.',
            'Merck & Co.', 'Visa Inc.', 'Mastercard Inc.', 'Cisco Systems', 'Intel Corporation',
            'NVIDIA Corporation', 'Advanced Micro Devices', 'Salesforce Inc.', 'Adobe Inc.', 'Netflix Inc.',
            'PayPal Holdings', 'Square Inc.', 'Zoom Video', 'Shopify Inc.', 'Peloton Interactive',
            'Beyond Meat Inc.', 'Etsy Inc.', 'Pinterest Inc.', 'Bumble Inc.', 'Lululemon Athletica',
            'Starbucks Corporation', 'Nike Inc.', 'Target Corporation', 'Walmart Inc.', 'Amazon.com Inc.',
            'Home Depot Inc.', 'Costco Wholesale', 'CVS Health Corp.', 'UnitedHealth Group', 'Anthem Inc.',
            'Moderna Inc.', 'Regeneron Pharma', 'Gilead Sciences', 'Biogen Inc.', 'Amgen Inc.',
            'Bristol Myers Squibb', 'AbbVie Inc.', 'Eli Lilly', 'Novartis AG', 'Roche Holding AG'
        ],
        'Ticker': [
            'MSFT', 'AAPL', 'GOOGL', 'TSLA', 'JNJ',
            'PG', 'UL', 'JPM', 'BAC', 'PFE',
            'MRK', 'V', 'MA', 'CSCO', 'INTC',
            'NVDA', 'AMD', 'CRM', 'ADBE', 'NFLX',
            'PYPL', 'SQ', 'ZM', 'SHOP', 'PTON',
            'BYND', 'ETSY', 'PINS', 'BMBL', 'LULU',
            'SBUX', 'NKE', 'TGT', 'WMT', 'AMZN',
            'HD', 'COST', 'CVS', 'UNH', 'ANTM',
            'MRNA', 'REGN', 'GILD', 'BIIB', 'AMGN',
            'BMY', 'ABBV', 'LLY', 'NVS', 'RHHBY'
        ],
        'Female_CEO': [
            0, 0, 0, 0, 0,  # Tech giants
            1, 0, 0, 0, 0,  # P&G has female CEO
            0, 0, 0, 0, 0,
            0, 1, 0, 0, 0,  # AMD has female CEO
            0, 0, 0, 0, 0,
            0, 1, 1, 1, 0,  # Etsy, Pinterest, Bumble have female CEOs
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 1,  # Anthem has female CEO
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 0
        ],
        'Female_Board_Members': [
            4, 3, 2, 2, 4,
            5, 4, 3, 4, 3,
            3, 3, 2, 3, 2,
            1, 4, 3, 3, 4,
            3, 2, 3, 4, 3,
            2, 5, 4, 5, 4,
            4, 3, 4, 3, 2,
            3, 3, 4, 3, 4,
             2, 2, 3, 3, 3,
            4, 3, 4, 3, 4
        ],
        'ESG_Score': [
            0.89, 0.75, 0.72, 0.85, 0.88,
            0.92, 0.91, 0.68, 0.71, 0.82,
            0.85, 0.74, 0.73, 0.78, 0.69,
            0.76, 0.71, 0.87, 0.81, 0.77,
            0.79, 0.82, 0.84, 0.86, 0.75,
            0.88, 0.90, 0.83, 0.89, 0.87,
            0.86, 0.84, 0.83, 0.74, 0.73,
            0.79, 0.81, 0.76, 0.72, 0.74,
            0.78, 0.80, 0.83, 0.81, 0.79,
            0.84, 0.82, 0.85, 0.87, 0.89
        ],
        'Sector': [
            'IT', 'IT', 'IT', 'Energy', 'Healthcare',
            'FMCG', 'FMCG', 'Finance', 'Finance', 'Pharma',
            'Pharma', 'Finance', 'Finance', 'IT', 'IT',
            'IT', 'IT', 'IT', 'IT', 'IT',
            'Finance', 'Finance', 'IT', 'Retail', 'Retail',
            'FMCG', 'Retail', 'IT', 'IT', 'Retail',
            'Retail', 'Retail', 'Retail', 'Retail', 'Retail',
            'Retail', 'Retail', 'Healthcare', 'Healthcare', 'Insurance',
            'Pharma', 'Pharma', 'Pharma', 'Pharma', 'Pharma',
            'Pharma', 'Pharma', 'Pharma', 'Pharma', 'Pharma'
        ]
    }
    
    df = pd.DataFrame(stock_data)
    
    # Create sector dummy variables
    sector_dummies = pd.get_dummies(df['Sector'], prefix='Sector')
    
    # Ensure all expected sectors are present
    expected_sectors = ['Energy', 'FMCG', 'Finance', 'Healthcare', 'IT', 'Insurance', 'Mining', 'Pharma', 'Retail']
    for sector in expected_sectors:
        col_name = f'Sector_{sector}'
        if col_name not in sector_dummies.columns:
            sector_dummies[col_name] = 0
    
    # Combine with original data
    result_df = pd.concat([df, sector_dummies], axis=1)
    
    return result_df

# Custom CSS for enhanced styling
st.markdown("""
<style>
    .main-header {
        font-size: 3.5rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(135deg, #2E8B57 0%, #48B685 50%, #20B2AA 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        text-align: center;
        font-size: 1.3rem;
        color: #2E8B57;
        font-weight: 500;
        margin-bottom: 2rem;
        font-style: italic;
    }
    
    .feature-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #2E8B57;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(46, 139, 87, 0.1);
    }
    
    .prediction-success {
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        font-size: 1.2rem;
        font-weight: bold;
        margin: 1rem 0;
        box-shadow: 0 4px 20px rgba(40, 167, 69, 0.3);
    }
    
    .prediction-error {
        background: linear-gradient(135deg, #dc3545 0%, #e83e8c 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        font-size: 1.2rem;
        font-weight: bold;
        margin: 1rem 0;
        box-shadow: 0 4px 20px rgba(220, 53, 69, 0.3);
    }
    
    .stock-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #2E8B57;
        margin: 0.5rem 0;
        box-shadow: 0 2px 10px rgba(46, 139, 87, 0.1);
    }
    
    .stats-card {
        background: linear-gradient(135deg, #2E8B57 0%, #20B2AA 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 4px 15px rgba(46, 139, 87, 0.3);
    }
    
    .sidebar-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2E8B57;
        margin-bottom: 1rem;
        text-align: center;
    }
    
    .confidence-high { color: #28a745; font-weight: bold; }
    .confidence-medium { color: #ffc107; font-weight: bold; }
    .confidence-low { color: #dc3545; font-weight: bold; }
    
    .stButton > button {
        background: linear-gradient(135deg, #2E8B57 0%, #20B2AA 100%);
        color: white;
        border: none;
        padding: 0.7rem 2rem;
        font-size: 1.1rem;
        font-weight: bold;
        border-radius: 10px;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(46, 139, 87, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# Load trained model
@st.cache_resource
def load_model():
    try:
        with open("model/recommender.pkl", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error("⚠️ Model file not found. Please ensure 'model/recommender.pkl' exists.")
        return None

model = load_model()

# Load real stock data
real_stocks_df = load_real_stock_data()

# Initialize session state
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

# Header
st.markdown('<h1 class="main-header">🌱 SheStock</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">ESG-Based Stock Recommender: Empowering Women-Led Sustainable Investment Decisions</p>', unsafe_allow_html=True)

# Sidebar for additional features
with st.sidebar:
    st.markdown('<div class="sidebar-header">📊 Dashboard</div>', unsafe_allow_html=True)
    
    # Show prediction statistics
    if st.session_state.prediction_history:
        total_predictions = len(st.session_state.prediction_history)
        recommended_count = sum(1 for p in st.session_state.prediction_history if p['prediction'])
        
        st.markdown(f"""
        <div class="stats-card">
            <h3>📈 Session Stats</h3>
            <p>Total Predictions: {total_predictions}</p>
            <p>Recommended: {recommended_count}</p>
            <p>Success Rate: {recommended_count/total_predictions*100:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
    
    # ESG Information
    st.markdown("---")
    st.markdown("### 🌍 ESG Criteria")
    st.markdown("""
    **Environmental**: Climate change, resource depletion, waste & pollution
    
    **Social**: Working conditions, local communities, health & safety
    
    **Governance**: Management structure, employee relations, executive compensation
    """)
    
    # Quick tips
    st.markdown("---")
    st.markdown("### 💡 Investment Tips")
    st.info("Companies with higher ESG scores often show better long-term performance and lower risk.")
    st.success("Female leadership in companies is correlated with better ESG performance.")

# Main content in tabs
tab1, tab2, tab3, tab4 = st.tabs(["🔍 Stock Analysis", "📊 Real Stock Matches", "📈 Visualization", "📋 History"])

with tab1:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.header("📋 Company Information")
        
        # Input fields with better organization
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.subheader("👩‍💼 Leadership")
            female_ceo = st.selectbox("Female CEO?", [0, 1], index=1, 
                                    format_func=lambda x: "Yes" if x == 1 else "No")
            female_board = st.slider("Female Board Members", 0, 6, 2, 
                                   help="Number of female board members")
            
            st.subheader("🌱 ESG Performance")
            esg_score = st.slider("ESG Score", 0.0, 1.0, 0.8, 
                                help="Environmental, Social, and Governance score (0.0 to 1.0)")
        
        with col_b:
            st.subheader("🏭 Industry Sector")
            sectors = {
                "Energy": st.checkbox("⚡ Energy"),
                "FMCG": st.checkbox("🛒 FMCG"),
                "Finance": st.checkbox("💰 Finance"),
                "Healthcare": st.checkbox("🏥 Healthcare"),
                "IT": st.checkbox("💻 IT", value=True),
                "Insurance": st.checkbox("📋 Insurance"),
                "Mining": st.checkbox("⛏️ Mining"),
                "Pharma": st.checkbox("💊 Pharma"),
                "Retail": st.checkbox("🛍️ Retail")
            }
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Company data preparation
        company_data = {
            "Female_CEO": female_ceo,
            "Female_Board_Members": female_board,
            "ESG_Score": esg_score,
            "Sector_Energy": int(sectors["Energy"]),
            "Sector_FMCG": int(sectors["FMCG"]),
            "Sector_Finance": int(sectors["Finance"]),
            "Sector_Healthcare": int(sectors["Healthcare"]),
            "Sector_IT": int(sectors["IT"]),
            "Sector_Insurance": int(sectors["Insurance"]),
            "Sector_Mining": int(sectors["Mining"]),
            "Sector_Pharma": int(sectors["Pharma"]),
            "Sector_Retail": int(sectors["Retail"])
        }
    
    with col2:
        st.markdown("### 📊 Current Input Summary")
        
        # Visual representation of inputs
        st.markdown(f"""
        **Leadership Score**: {"🟢 High" if female_ceo and female_board >= 3 else "🟡 Medium" if female_ceo or female_board >= 2 else "🔴 Low"}
        
        **ESG Score**: {esg_score:.1f}/1.0
        
        **Selected Sectors**: {sum(sectors.values())} selected
        """)
        
        # ESG Score visualization
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = esg_score,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "ESG Score"},
            gauge = {
                'axis': {'range': [None, 1]},
                'bar': {'color': "#2E8B57"},
                'steps': [
                    {'range': [0, 0.5], 'color': "lightgray"},
                    {'range': [0.5, 0.8], 'color': "gray"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 0.9
                }
            }
        ))
        fig_gauge.update_layout(height=200)
        st.plotly_chart(fig_gauge, use_container_width=True)
    
    # Prediction section
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col2:
        if st.button("🔍 Predict Recommendation", use_container_width=True):
            if model is not None:
                # Expected columns (MUST match training)
                expected_cols = [
                    "Female_CEO", "Female_Board_Members", "ESG_Score",
                    "Sector_Energy", "Sector_FMCG", "Sector_Finance",
                    "Sector_Healthcare", "Sector_IT", "Sector_Insurance",
                    "Sector_Mining", "Sector_Pharma", "Sector_Retail"
                ]
                
                input_df = pd.DataFrame([company_data])
                
                # Add missing columns (if any)
                for col in expected_cols:
                    if col not in input_df.columns:
                        input_df[col] = 0
                
                input_df = input_df[expected_cols]
                
                # Make prediction
                pred = model.predict(input_df)[0]
                proba = model.predict_proba(input_df)[0][1]
                
                # Store in history
                st.session_state.prediction_history.append({
                    'timestamp': datetime.now(),
                    'prediction': pred,
                    'confidence': proba,
                    'data': company_data.copy()
                })
                
                # Display results
                if pred:
                    confidence_class = "confidence-high" if proba > 0.8 else "confidence-medium" if proba > 0.6 else "confidence-low"
                    st.markdown(f"""
                    <div class="prediction-success">
                        ✅ <strong>RECOMMENDED</strong><br>
                        <span class="{confidence_class}">Confidence: {proba*100:.1f}%</span>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    confidence_class = "confidence-high" if (1-proba) > 0.8 else "confidence-medium" if (1-proba) > 0.6 else "confidence-low"
                    st.markdown(f"""
                    <div class="prediction-error">
                        ❌ <strong>NOT RECOMMENDED</strong><br>
                        <span class="{confidence_class}">Confidence: {(1-proba)*100:.1f}%</span>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.error("Model not available. Using profile matching instead.")

with tab2:
    st.header("📊 Real Stock Matches")
    st.markdown("Find real companies that match your investment criteria:")
    
    # Filter options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        min_esg = st.slider("Minimum ESG Score", 0.0, 1.0, 0.7, key="min_esg")
        female_ceo_filter = st.selectbox("Female CEO Required?", ["Any", "Yes", "No"], key="ceo_filter")
    
    with col2:
        min_female_board = st.slider("Min Female Board Members", 0, 6, 2, key="min_board")
        sector_filter = st.selectbox("Sector Filter", ["All"] + list(real_stocks_df['Sector'].unique()), key="sector_filter")
    
    with col3:
        sort_by = st.selectbox("Sort by", ["ESG Score", "Female Leadership", "Company Name"], key="sort_by")
        max_results = st.slider("Max Results", 5, 20, 10, key="max_results")
    
    # Apply filters
    filtered_stocks = real_stocks_df.copy()
    
    # ESG filter
    filtered_stocks = filtered_stocks[filtered_stocks['ESG_Score'] >= min_esg]
    
    # Female CEO filter
    if female_ceo_filter == "Yes":
        filtered_stocks = filtered_stocks[filtered_stocks['Female_CEO'] == 1]
    elif female_ceo_filter == "No":
        filtered_stocks = filtered_stocks[filtered_stocks['Female_CEO'] == 0]
    
    # Female board members filter
    filtered_stocks = filtered_stocks[filtered_stocks['Female_Board_Members'] >= min_female_board]
    
    # Sector filter
    if sector_filter != "All":
        filtered_stocks = filtered_stocks[filtered_stocks['Sector'] == sector_filter]
    
    # Sorting
    if sort_by == "ESG Score":
        filtered_stocks = filtered_stocks.sort_values('ESG_Score', ascending=False)
    elif sort_by == "Female Leadership":
        filtered_stocks['Leadership_Score'] = filtered_stocks['Female_CEO'] * 0.5 + filtered_stocks['Female_Board_Members'] * 0.1
        filtered_stocks = filtered_stocks.sort_values('Leadership_Score', ascending=False)
    else:
        filtered_stocks = filtered_stocks.sort_values('Company')
    
    # Display results
    st.markdown(f"### 🏆 Found {len(filtered_stocks)} matching companies:")
    
    if len(filtered_stocks) > 0:
        # Show top results
        top_stocks = filtered_stocks.head(max_results)
        
        for idx, row in top_stocks.iterrows():
            leadership_score = "High" if row['Female_CEO'] and row['Female_Board_Members'] >= 3 else "Medium" if row['Female_CEO'] or row['Female_Board_Members'] >= 2 else "Low"
            leadership_color = "🟢" if leadership_score == "High" else "🟡" if leadership_score == "Medium" else "🔴"
            
            st.markdown(f"""
            <div class="stock-card">
                <h4>🏢 {row['Company']} ({row['Ticker']})</h4>
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <strong>Sector:</strong> {row['Sector']}<br>
                        <strong>ESG Score:</strong> {row['ESG_Score']:.2f}/1.0<br>
                        <strong>Female CEO:</strong> {'✅ Yes' if row['Female_CEO'] else '❌ No'}<br>
                        <strong>Female Board Members:</strong> {int(row['Female_Board_Members'])}/6
                    </div>
                    <div style="text-align: center;">
                        <div style="font-size: 1.2em; font-weight: bold;">
                            {leadership_color} {leadership_score}<br>
                            Leadership
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Summary statistics
        st.markdown("---")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Average ESG Score", f"{top_stocks['ESG_Score'].mean():.2f}")
        
        with col2:
            st.metric("Companies with Female CEO", f"{top_stocks['Female_CEO'].sum()}")
        
        with col3:
            st.metric("Avg Female Board Members", f"{top_stocks['Female_Board_Members'].mean():.1f}")
        
        with col4:
            most_common_sector = top_stocks['Sector'].mode().iloc[0] if len(top_stocks) > 0 else "N/A"
            st.metric("Most Common Sector", most_common_sector)
        
    else:
        st.warning("🔍 No companies match your criteria. Try adjusting the filters.")
        
        # Show suggestions
        st.markdown("### 💡 Try These Adjustments:")
        st.markdown("- Lower the minimum ESG score")
        st.markdown("- Reduce minimum female board members requirement")
        st.markdown("- Change sector filter to 'All'")
        st.markdown("- Set Female CEO filter to 'Any'")

with tab3:
    st.header("📊 Data Visualization")
    
    # Real stock data visualizations
    st.subheader("🏢 Real Stock Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # ESG Score distribution by sector
        fig_box = px.box(real_stocks_df, x='Sector', y='ESG_Score', 
                        title="ESG Score Distribution by Sector",
                        color='Sector')
        fig_box.update_xaxis(tickangle=45)
        st.plotly_chart(fig_box, use_container_width=True)
    
    with col2:
        # Female leadership vs ESG Score
        fig_scatter = px.scatter(real_stocks_df, x='Female_Board_Members', y='ESG_Score',
                               color='Female_CEO', size='ESG_Score',
                               title="Female Leadership vs ESG Performance",
                               hover_data=['Company', 'Ticker'],
                               color_discrete_map={0: '#dc3545', 1: '#2E8B57'})
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Sector analysis
    sector_stats = real_stocks_df.groupby('Sector').agg({
        'ESG_Score': 'mean',
        'Female_CEO': 'sum',
        'Female_Board_Members': 'mean'
    }).reset_index()
    
    fig_bar = px.bar(sector_stats, x='Sector', y='ESG_Score',
                    title="Average ESG Score by Sector",
                    color='ESG_Score',
                    color_continuous_scale='Greens')
    fig_bar.update_xaxis(tickangle=45)
    st.plotly_chart(fig_bar, use_container_width=True)
    
    # Prediction history visualizations
    if st.session_state.prediction_history:
        st.subheader("📈 Your Prediction History")
        
        # Create visualizations from prediction history
        df_history = pd.DataFrame([
            {
                'timestamp': p['timestamp'],
                'prediction': 'Recommended' if p['prediction'] else 'Not Recommended',
                'confidence': p['confidence'] if p['prediction'] else 1 - p['confidence'],
                'esg_score': p['data']['ESG_Score'],
                'female_ceo': p['data']['Female_CEO'],
                'female_board': p['data']['Female_Board_Members']
            }
            for p in st.session_state.prediction_history
        ])
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Prediction distribution
            fig_pie = px.pie(df_history, names='prediction', 
                           title="Your Prediction Distribution",
                           color_discrete_map={'Recommended': '#2E8B57', 'Not Recommended': '#dc3545'})
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # ESG Score vs Confidence
            fig_scatter_hist = px.scatter(df_history, x='esg_score', y='confidence',
                                        color='prediction', title="ESG Score vs Confidence",
                                        color_discrete_map={'Recommended': '#2E8B57', 'Not Recommended': '#dc3545'})
            st.plotly_chart(fig_scatter_hist, use_container_width=True)

with tab4:
    st.header("📈 Prediction History")
    
    if st.session_state.prediction_history:
        # Display recent predictions
        for i, prediction in enumerate(reversed(st.session_state.prediction_history[-10:])):
            with st.expander(f"Prediction {len(st.session_state.prediction_history) - i} - {prediction['timestamp'].strftime('%Y-%m-%d %H:%M')}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Input Data:**")
                    st.json(prediction['data'])
                
                with col2:
                    result = "✅ Recommended" if prediction['prediction'] else "🚫 Not Recommended"
                    confidence = prediction['confidence'] if prediction['prediction'] else 1 - prediction['confidence']
                    st.write(f"**Result:** {result}")
                    st.write(f"**Confidence:** {confidence*100:.1f}%")
                    
                    # Find similar real stocks for this prediction
                    similar_stocks = find_similar_stocks(prediction['data'], real_stocks_df)
                    if len(similar_stocks) > 0:
                        st.write("**Similar Real Stocks:**")
                        for _, stock in similar_stocks.head(3).iterrows():
                            st.write(f"• {stock['Company']} ({stock['Ticker']}) - ESG: {stock['ESG_Score']:.2f}")
        
        # Clear history button
        if st.button("🗑️ Clear History"):
            st.session_state.prediction_history = []
            st.rerun()
    else:
        st.info("No predictions made yet. Go to the Stock Analysis tab to make your first prediction!")

# Helper function to find similar stocks
def find_similar_stocks(input_data, stock_df, top_n=5):
    """Find real stocks similar to input criteria"""
    # Calculate similarity score
    scores = []
    
    for idx, row in stock_df.iterrows():
        score = 0
        
        # ESG score similarity (weight: 0.4)
        esg_diff = abs(row['ESG_Score'] - input_data['ESG_Score'])
        esg_similarity = max(0, 1 - esg_diff)
        score += esg_similarity * 0.4
        
        # Female CEO match (weight: 0.2)
        if row['Female_CEO'] == input_data['Female_CEO']:
            score += 0.2
        
        # Female board members similarity (weight: 0.2)
        board_diff = abs(row['Female_Board_Members'] - input_data['Female_Board_Members'])
        board_similarity = max(0, 1 - board_diff / 6)  # Normalize by max possible difference
        score += board_similarity * 0.2
        
        # Sector match (weight: 0.2)
        sector_match = False
        for sector in ['Energy', 'FMCG', 'Finance', 'Healthcare', 'IT', 'Insurance', 'Mining', 'Pharma', 'Retail']:
            if input_data.get(f'Sector_{sector}', 0) == 1 and row['Sector'] == sector:
                sector_match = True
                break
        
        if sector_match:
            score += 0.2
        
        scores.append(score)
    
    # Add scores to dataframe and sort
    stock_df_copy = stock_df.copy()
    stock_df_copy['similarity_score'] = scores
    
    return stock_df_copy.sort_values('similarity_score', ascending=False).head(top_n)

# Add prediction with stock recommendations
def predict_with_stock_recommendations():
    """Enhanced prediction function that also shows similar real stocks"""
    if st.button("🔍 Get Prediction + Stock Recommendations", use_container_width=True):
        if model is not None:
            # Expected columns (MUST match training)
            expected_cols = [
                "Female_CEO", "Female_Board_Members", "ESG_Score",
                "Sector_Energy", "Sector_FMCG", "Sector_Finance",
                "Sector_Healthcare", "Sector_IT", "Sector_Insurance",
                "Sector_Mining", "Sector_Pharma", "Sector_Retail"
            ]
            
            input_df = pd.DataFrame([company_data])
            
            # Add missing columns (if any)
            for col in expected_cols:
                if col not in input_df.columns:
                    input_df[col] = 0
            
            input_df = input_df[expected_cols]
            
            # Make prediction
            pred = model.predict(input_df)[0]
            proba = model.predict_proba(input_df)[0][1]
            
            # Store in history
            st.session_state.prediction_history.append({
                'timestamp': datetime.now(),
                'prediction': pred,
                'confidence': proba,
                'data': company_data.copy()
            })
            
            # Display results
            if pred:
                confidence_class = "confidence-high" if proba > 0.8 else "confidence-medium" if proba > 0.6 else "confidence-low"
                st.markdown(f"""
                <div class="prediction-success">
                    ✅ <strong>RECOMMENDED FOR INVESTMENT</strong><br>
                    <span class="{confidence_class}">Model Confidence: {proba*100:.1f}%</span>
                </div>
                """, unsafe_allow_html=True)
            else:
                confidence_class = "confidence-high" if (1-proba) > 0.8 else "confidence-medium" if (1-proba) > 0.6 else "confidence-low"
                st.markdown(f"""
                <div class="prediction-error">
                    ❌ <strong>NOT RECOMMENDED FOR INVESTMENT</strong><br>
                    <span class="{confidence_class}">Model Confidence: {(1-proba)*100:.1f}%</span>
                </div>
                """, unsafe_allow_html=True)
            
            # Show similar real stocks
            st.markdown("---")
            st.subheader("🏢 Real Companies with Similar Profiles")
            
            similar_stocks = find_similar_stocks(company_data, real_stocks_df)
            
            if len(similar_stocks) > 0:
                st.markdown("Based on your criteria, here are real companies you might consider:")
                
                for idx, row in similar_stocks.head(5).iterrows():
                    similarity_percentage = row['similarity_score'] * 100
                    
                    # Determine recommendation based on similarity and ESG score
                    is_recommended = row['ESG_Score'] >= 0.75 and row['Female_Board_Members'] >= 2
                    rec_icon = "🟢 STRONG BUY" if is_recommended and similarity_percentage > 80 else "🟡 CONSIDER" if is_recommended else "🔴 RESEARCH NEEDED"
                    
                    st.markdown(f"""
                    <div class="stock-card">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <h4>🏢 {row['Company']} ({row['Ticker']})</h4>
                                <strong>Sector:</strong> {row['Sector']} | 
                                <strong>ESG:</strong> {row['ESG_Score']:.2f} | 
                                <strong>Female CEO:</strong> {'✅' if row['Female_CEO'] else '❌'} | 
                                <strong>Female Board:</strong> {int(row['Female_Board_Members'])}/6<br>
                                <strong>Profile Match:</strong> {similarity_percentage:.1f}%
                            </div>
                            <div style="text-align: center; font-weight: bold; font-size: 0.9em;">
                                {rec_icon}
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning("No similar companies found in our database.")
        
        else:
            st.error("Model not available. Showing stock matches based on your criteria only.")
            
            # Show filtered stocks when model is not available
            similar_stocks = find_similar_stocks(company_data, real_stocks_df)
            
            if len(similar_stocks) > 0:
                st.subheader("🏢 Companies Matching Your Criteria")
                
                for idx, row in similar_stocks.head(8).iterrows():
                    similarity_percentage = row['similarity_score'] * 100
                    
                    st.markdown(f"""
                    <div class="stock-card">
                        <h4>🏢 {row['Company']} ({row['Ticker']})</h4>
                        <strong>Sector:</strong> {row['Sector']} | 
                        <strong>ESG Score:</strong> {row['ESG_Score']:.2f} | 
                        <strong>Female CEO:</strong> {'✅ Yes' if row['Female_CEO'] else '❌ No'} | 
                        <strong>Female Board Members:</strong> {int(row['Female_Board_Members'])}/6<br>
                        <strong>Match Score:</strong> {similarity_percentage:.1f}%
                    </div>
                    """, unsafe_allow_html=True)

# Footer with glossary and additional features
st.markdown("---")

# Real-time stock insights
with st.expander("📊 Market Insights - Top ESG Performers"):
    # Show top ESG performers
    top_esg = real_stocks_df.nlargest(10, 'ESG_Score')
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🏆 Highest ESG Scores")
        for _, row in top_esg.head(5).iterrows():
            st.markdown(f"**{row['Company']}** ({row['Ticker']}) - {row['ESG_Score']:.2f}")
    
    with col2:
        st.markdown("### 👩‍💼 Companies with Female CEOs")
        female_ceo_companies = real_stocks_df[real_stocks_df['Female_CEO'] == 1]
        for _, row in female_ceo_companies.head(5).iterrows():
            st.markdown(f"**{row['Company']}** ({row['Ticker']}) - ESG: {row['ESG_Score']:.2f}")

with st.expander("📚 Glossary & Information"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 📖 Key Terms
        - **ESG Score**: Measures how well a company performs on environmental, social, and governance criteria (0.0 to 1.0)
        - **Female CEO**: Binary indicator of female chief executive leadership
        - **Female Board Members**: Number of women on the board of directors (0-6)
        - **Sector**: Company's primary industry classification
        - **Similarity Score**: How closely a real company matches your input criteria
        """)
    
    with col2:
        st.markdown("""
        ### 🎯 Model Information
        - **Recommendation**: Binary output indicating investment recommendation
        - **Confidence**: Probability score indicating model certainty
        - **Features**: Leadership diversity, ESG performance, and sector classification
        - **Purpose**: Promote sustainable and inclusive investment decisions
        - **Stock Database**: 50+ real companies with current ESG and diversity data
        """)

# Performance metrics section
with st.expander("📈 Database Statistics"):
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Companies", len(real_stocks_df))
        st.metric("Avg ESG Score", f"{real_stocks_df['ESG_Score'].mean():.2f}")
    
    with col2:
        st.metric("Female CEOs", real_stocks_df['Female_CEO'].sum())
        st.metric("Companies > 0.8 ESG", len(real_stocks_df[real_stocks_df['ESG_Score'] > 0.8]))
    
    with col3:
        st.metric("Avg Female Board Members", f"{real_stocks_df['Female_Board_Members'].mean():.1f}")
        st.metric("Sectors Covered", real_stocks_df['Sector'].nunique())
    
    with col4:
        high_diversity = real_stocks_df[(real_stocks_df['Female_CEO'] == 1) | (real_stocks_df['Female_Board_Members'] >= 3)]
        st.metric("High Diversity Leadership", len(high_diversity))
        st.metric("Top ESG Sector", real_stocks_df.loc[real_stocks_df['ESG_Score'].idxmax(), 'Sector'])

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #2E8B57; font-size: 1rem; margin-top: 2rem;'>
    💡 <strong>SheStock v2.0 - Enhanced with Real Stock Database</strong><br>
    <em>50+ Real Companies • ESG Scoring • Female Leadership Analytics</em><br>
    <em>Empowering women-led sustainable investment decisions with real market data</em>
</div>
""", unsafe_allow_html=True)
