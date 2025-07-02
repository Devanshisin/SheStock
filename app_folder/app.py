

# import streamlit as st
# import pandas as pd
# import pickle

# # Set page config
# st.set_page_config(page_title="SheStock Recommender", layout="centered")

# # Load trained model
# with open("model/recommender.pkl", "rb") as f:
#     model = pickle.load(f)

# st.title("📈 SheStock: ESG-Based Stock Recommender")
# st.markdown("Analyze ESG and leadership data to predict if a stock is 💹 **Recommended** or 🚫 **Not Recommended**.")

# # Input fields
# st.header("📋 Company Information")
# company_data = {
#     "Female_CEO": st.selectbox("👩‍💼 Female CEO?", [0, 1], index=1),
#     "Female_Board_Members": st.slider("🧑‍🤝‍🧑 Female Board Members", 0, 6, 2),
#     "ESG_Score": st.slider("🌱 ESG Score (0.0 to 1.0)", 0.0, 1.0, 0.8),
#     "Sector_Energy": st.checkbox("⚡ Energy"),
#     "Sector_FMCG": st.checkbox("🛒 FMCG"),
#     "Sector_Finance": st.checkbox("💰 Finance"),
#     "Sector_Healthcare": st.checkbox("🏥 Healthcare"),
#     "Sector_IT": st.checkbox("💻 IT", value=True),
#     "Sector_Insurance": st.checkbox("📋 Insurance"),
#     "Sector_Mining": st.checkbox("⛏️ Mining"),
#     "Sector_Pharma": st.checkbox("💊 Pharma"),
#     "Sector_Retail": st.checkbox("🛍️ Retail"),
# }

# # Expected columns (MUST match training)
# expected_cols = [
#     "Female_CEO",
#     "Female_Board_Members",
#     "ESG_Score",
#     "Sector_Energy",
#     "Sector_FMCG",
#     "Sector_Finance",
#     "Sector_Healthcare",
#     "Sector_IT",
#     "Sector_Insurance",
#     "Sector_Mining",
#     "Sector_Pharma",
#     "Sector_Retail"
# ]

# # Prediction
# if st.button("🔍 Predict Recommendation"):
#     input_df = pd.DataFrame([company_data])

#     # Add missing columns (if any)
#     for col in expected_cols:
#         if col not in input_df.columns:
#             input_df[col] = 0

#     input_df = input_df[expected_cols]

#     pred = model.predict(input_df)[0]
#     proba = model.predict_proba(input_df)[0][1]

#     if pred:
#         st.success(f"✅ Recommended with {proba*100:.2f}% confidence")
#     else:
#         st.error(f"🚫 Not Recommended with {(1-proba)*100:.2f}% confidence")

# # Glossary (Optional)
# with st.expander("📚 Glossary"):
#     st.markdown("""
#     - **ESG Score**: Measures how well a company performs on environmental, social, and governance criteria.
#     - **Female CEO / Board Members**: Indicates presence of women in leadership.
#     - **Sector**: Company’s industry classification.
#     - **Recommendation**: Model’s output whether to recommend stock or not based on features.
#     """)

# st.caption("💡 Created with ❤️ by Devanshi for ESG-conscious investing.")


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
tab1, tab2, tab3 = st.tabs(["🔍 Stock Analysis", "📊 Visualization", "📈 History"])

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
                # else:
                #     confidence_class = "confidence-high" if (1-proba) > 0.8 else "confidence-medium" if (1-proba) > 0.6 else "confidence-low"
                #     st.markdown(f"""
                    <div class="prediction-error">
                       <strong> &#128683; NOT RECOMMENDED</strong><br>

                        <span class="{confidence_class}">Confidence: {(1-proba)*100:.1f}%</span>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.error("Model not available. Please check the model file.")
                --- Show real similar companies ---
                try:
                    esg_df = pd.read_csv("preprocessed_esg_dataset.csv")
                    # Filter dataset for similar profile
                    matching = esg_df[
                        (esg_df["Female_CEO"] == female_ceo) &
                        (esg_df["Female_Board_Members"] >= female_board - 1) &
                        (esg_df["Female_Board_Members"] <= female_board + 1) &
                        (esg_df["ESG_Score"] >= esg_score - 0.1) &
                        (esg_df["ESG_Score"] <= esg_score + 0.1)
                    ]

                    # Sort by ESG Score + Board representation
                    matching["Score"] = matching["ESG_Score"] + 0.05 * matching["Female_Board_Members"]
                    top_stocks = matching.sort_values(by="Score", ascending=False).head(5)

                    if not top_stocks.empty:
                        st.markdown("### 🏆 Top Real Stocks Matching Profile")
                        for i, row in top_stocks.iterrows():
                            st.markdown(f"""
                            **{row['Company']}** ({row['Ticker']})  
                            ESG Score: {row['ESG_Score']:.2f} | Female Board Members: {int(row['Female_Board_Members'])}
                            """)
                    else:
                        st.warning("No exact real stock matches found. Try changing the ESG or leadership values.")
                except Exception as e:
                    st.error(f"❌ Could not load real stock data: {e}")

with tab2:
    st.header("📊 Data Visualization")
    
    if st.session_state.prediction_history:
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
                           title="Prediction Distribution",
                           color_discrete_map={'Recommended': '#2E8B57', 'Not Recommended': '#dc3545'})
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # ESG Score vs Confidence
            fig_scatter = px.scatter(df_history, x='esg_score', y='confidence',
                                   color='prediction', title="ESG Score vs Confidence",
                                   color_discrete_map={'Recommended': '#2E8B57', 'Not Recommended': '#dc3545'})
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Leadership impact analysis
        fig_bar = px.bar(df_history.groupby(['female_ceo', 'prediction']).size().reset_index(name='count'),
                        x='female_ceo', y='count', color='prediction',
                        title="Impact of Female CEO on Recommendations",
                        color_discrete_map={'Recommended': '#2E8B57', 'Not Recommended': '#dc3545'})
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("📈 Make some predictions first to see visualizations!")
        
        # Show sample visualization
        sample_data = pd.DataFrame({
            'ESG_Score': np.random.uniform(0.3, 0.95, 50),
            'Female_Leadership': np.random.choice(['High', 'Medium', 'Low'], 50),
            'Recommendation': np.random.choice(['Recommended', 'Not Recommended'], 50)
        })
        
        fig_sample = px.scatter(sample_data, x='ESG_Score', y='Female_Leadership',
                              color='Recommendation', title="Sample: ESG Score vs Female Leadership",
                              color_discrete_map={'Recommended': '#2E8B57', 'Not Recommended': '#dc3545'})
        st.plotly_chart(fig_sample, use_container_width=True)

with tab3:
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
        
        # Clear history button
        if st.button("🗑️ Clear History"):
            st.session_state.prediction_history = []
            st.rerun()
    else:
        st.info("No predictions made yet. Go to the Stock Analysis tab to make your first prediction!")

# Footer with glossary
st.markdown("---")
with st.expander("📚 Glossary & Information"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 📖 Key Terms
        - **ESG Score**: Measures how well a company performs on environmental, social, and governance criteria (0.0 to 1.0)
        - **Female CEO**: Binary indicator of female chief executive leadership
        - **Female Board Members**: Number of women on the board of directors (0-6)
        - **Sector**: Company's primary industry classification
        """)
    
    with col2:
        st.markdown("""
        ### 🎯 Model Information
        - **Recommendation**: Binary output indicating investment recommendation
        - **Confidence**: Probability score indicating model certainty
        - **Features**: Leadership diversity, ESG performance, and sector classification
        - **Purpose**: Promote sustainable and inclusive investment decisions
        """)

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #2E8B57; font-size: 1rem; margin-top: 2rem;'>
    💡 <strong>Created with ❤️ by Devanshi for ESG-conscious investing</strong><br>
    <em>Empowering women-led sustainable investment decisions</em>
</div>
""", unsafe_allow_html=True)
