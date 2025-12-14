import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
import warnings
import os
from pathlib import Path
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION & STYLING
# =============================================================================

# Professional color palette
COLORS = {
    'primary': '#1f77b4',      # Blue
    'secondary': '#ff7f0e',    # Orange
    'success': '#2ca02c',      # Green
    'danger': '#d62728',       # Red
    'warning': '#ff9800',      # Amber
    'info': '#17a2b8',         # Cyan
    'purple': '#9467bd',       # Purple
    'pink': '#e377c2',         # Pink
    'brown': '#8c564b',        # Brown
    'gray': '#7f7f7f',         # Gray
    'olive': '#bcbd22',        # Olive
    'teal': '#17becf'          # Teal
}

SEQUENTIAL_COLORS = ['#08519c', '#3182bd', '#6baed6', '#9ecae1', '#c6dbef', '#eff3ff']
DIVERGING_COLORS = ['#d73027', '#fc8d59', '#fee090', '#e0f3f8', '#91bfdb', '#4575b4']

st.set_page_config(
    page_title="NovaMart Marketing Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stMetric {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: white;
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
    }
    h1 {
        color: #1f77b4;
        font-weight: 600;
    }
    h2, h3 {
        color: #2c3e50;
    }
    </style>
""", unsafe_allow_html=True)

# =============================================================================
# DATA LOADING - FLEXIBLE PATH HANDLING
# =============================================================================

def find_data_directory():
    """
    Try to find the data directory in multiple possible locations
    """
    possible_paths = [
        # Current directory
        Path("."),
        Path("./data"),
        Path("./marketing_dataset"),
        # Parent directory
        Path(".."),
        Path("../data"),
        Path("../marketing_dataset"),
        # Common structures
        Path("./NovaMart_Marketing_Analytics_Dataset/marketing_dataset"),
        Path("../NovaMart_Marketing_Analytics_Dataset/marketing_dataset"),
    ]
    
    for path in possible_paths:
        if path.exists() and (path / "campaign_performance.csv").exists():
            return path
    
    return None

@st.cache_data
def load_data(data_path=None):
    """
    Load all datasets with flexible path handling and file upload fallback
    """
    data = {}
    
    # If no path provided, try to find it
    if data_path is None:
        data_path = find_data_directory()
    
    # If still no path found, return None to trigger upload
    if data_path is None:
        return None
    
    try:
        data['campaigns'] = pd.read_csv(Path(data_path) / "campaign_performance.csv", parse_dates=['date'])
        data['customers'] = pd.read_csv(Path(data_path) / "customer_data.csv")
        data['products'] = pd.read_csv(Path(data_path) / "product_sales.csv")
        data['leads'] = pd.read_csv(Path(data_path) / "lead_scoring_results.csv")
        data['feature_importance'] = pd.read_csv(Path(data_path) / "feature_importance.csv")
        data['learning_curve'] = pd.read_csv(Path(data_path) / "learning_curve.csv")
        data['geographic'] = pd.read_csv(Path(data_path) / "geographic_data.csv")
        data['attribution'] = pd.read_csv(Path(data_path) / "channel_attribution.csv")
        data['funnel'] = pd.read_csv(Path(data_path) / "funnel_data.csv")
        data['journey'] = pd.read_csv(Path(data_path) / "customer_journey.csv")
        data['correlation'] = pd.read_csv(Path(data_path) / "correlation_matrix.csv", index_col=0)
        
        return data
    except FileNotFoundError as e:
        st.error(f"❌ Data file not found: {e}")
        return None
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        return None

def load_data_from_uploads(uploaded_files):
    """
    Load data from uploaded files
    """
    data = {}
    
    file_mapping = {
        'campaign_performance.csv': 'campaigns',
        'customer_data.csv': 'customers',
        'product_sales.csv': 'products',
        'lead_scoring_results.csv': 'leads',
        'feature_importance.csv': 'feature_importance',
        'learning_curve.csv': 'learning_curve',
        'geographic_data.csv': 'geographic',
        'channel_attribution.csv': 'attribution',
        'funnel_data.csv': 'funnel',
        'customer_journey.csv': 'journey',
        'correlation_matrix.csv': 'correlation'
    }
    
    for uploaded_file in uploaded_files:
        if uploaded_file.name in file_mapping:
            key = file_mapping[uploaded_file.name]
            try:
                if key == 'campaigns':
                    data[key] = pd.read_csv(uploaded_file, parse_dates=['date'])
                elif key == 'correlation':
                    data[key] = pd.read_csv(uploaded_file, index_col=0)
                else:
                    data[key] = pd.read_csv(uploaded_file)
            except Exception as e:
                st.error(f"Error loading {uploaded_file.name}: {e}")
    
    # Check if all required files are loaded
    required_keys = list(file_mapping.values())
    missing = [k for k in required_keys if k not in data]
    
    if missing:
        st.warning(f"⚠️ Missing files: {', '.join([k + '.csv' for k in missing])}")
        return None
    
    return data

def show_file_uploader():
    """
    Display file uploader interface
    """
    st.title("📁 Data Upload Required")
    st.markdown("""
    ### Please upload the following CSV files:
    
    Upload all 11 CSV files from your dataset:
    1. `campaign_performance.csv`
    2. `customer_data.csv`
    3. `product_sales.csv`
    4. `lead_scoring_results.csv`
    5. `feature_importance.csv`
    6. `learning_curve.csv`
    7. `geographic_data.csv`
    8. `channel_attribution.csv`
    9. `funnel_data.csv`
    10. `customer_journey.csv`
    11. `correlation_matrix.csv`
    """)
    
    uploaded_files = st.file_uploader(
        "Choose CSV files",
        type=['csv'],
        accept_multiple_files=True,
        help="Select all 11 CSV files from your dataset"
    )
    
    if uploaded_files:
        st.info(f"📊 {len(uploaded_files)} files uploaded")
        
        # Show uploaded file names
        with st.expander("View uploaded files"):
            for f in uploaded_files:
                st.text(f"✓ {f.name}")
        
        if len(uploaded_files) >= 11:
            return load_data_from_uploads(uploaded_files)
    
    return None

# =============================================================================
# SIDEBAR NAVIGATION
# =============================================================================
def sidebar():
    """Enhanced sidebar with filters"""
    st.sidebar.title("📊 NovaMart Analytics")
    st.sidebar.markdown("---")
    
    page = st.sidebar.radio(
        "Navigate to:",
        [
            "🏠 Executive Overview",
            "📈 Campaign Analytics",
            "👥 Customer Insights",
            "📦 Product Performance",
            "🗺️ Geographic Analysis",
            "🎯 Attribution & Funnel",
            "🛤️ Customer Journey",
            "🤖 ML Model Evaluation"
        ]
    )
    
    st.sidebar.markdown("---")
    
    # Global filters
    with st.sidebar.expander("🔧 Global Filters", expanded=False):
        st.info("Apply filters across all pages")
        date_filter = st.checkbox("Enable Date Filter")
        region_filter = st.checkbox("Enable Region Filter")
    
    st.sidebar.markdown("---")
    st.sidebar.info("**Masters of AI in Business**\n\nData Visualization Assignment\n\n📅 2024")
    
    return page

# =============================================================================
# PAGE FUNCTIONS (keeping all your existing page functions)
# =============================================================================

def page_executive_overview(data):
    """Executive Overview Dashboard with KPIs and trends"""
    st.title("🏠 Executive Overview")
    st.markdown("### Key Performance Metrics at a Glance")
    
    campaigns = data['campaigns']
    customers = data['customers']
    products = data['products']
    
    # KPI Cards with deltas
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        total_revenue = campaigns['revenue'].sum()
        st.metric(
            "Total Revenue", 
            f"₹{total_revenue/1e7:.2f} Cr",
            delta=f"{campaigns.groupby('quarter')['revenue'].sum().pct_change().iloc[-1]*100:.1f}% QoQ"
        )
    
    with col2:
        total_conversions = campaigns['conversions'].sum()
        st.metric(
            "Total Conversions", 
            f"{total_conversions:,}",
            delta=f"{campaigns.groupby('quarter')['conversions'].sum().pct_change().iloc[-1]*100:.1f}% QoQ"
        )
    
    with col3:
        avg_roas = campaigns[campaigns['roas'] > 0]['roas'].mean()
        st.metric("Avg ROAS", f"{avg_roas:.2f}x", delta="Healthy")
    
    with col4:
        total_customers = len(customers)
        churned = customers['is_churned'].sum()
        st.metric(
            "Active Customers", 
            f"{total_customers - churned:,}",
            delta=f"-{churned:,} churned"
        )
    
    with col5:
        avg_satisfaction = customers['satisfaction_score'].mean()
        st.metric(
            "Avg Satisfaction", 
            f"{avg_satisfaction:.2f}/5",
            delta=f"{(avg_satisfaction - 3) / 3 * 100:.1f}% vs baseline"
        )
    
    st.markdown("---")
    
    # Revenue Trend with Moving Average
    st.subheader("📈 Revenue Trend Over Time")
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        aggregation = st.selectbox(
            "Aggregation Period",
            ["Daily", "Weekly", "Monthly"],
            index=2
        )
        show_ma = st.checkbox("Show Moving Average", value=True)
    
    with col1:
        freq_map = {"Daily": "D", "Weekly": "W", "Monthly": "M"}
        revenue_trend = campaigns.groupby(pd.Grouper(key='date', freq=freq_map[aggregation]))['revenue'].sum().reset_index()
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=revenue_trend['date'],
            y=revenue_trend['revenue'],
            mode='lines+markers',
            name='Revenue',
            line=dict(color=COLORS['primary'], width=2),
            marker=dict(size=6),
            hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Revenue: ₹%{y:,.0f}<extra></extra>'
        ))
        
        if show_ma and len(revenue_trend) > 7:
            revenue_trend['ma'] = revenue_trend['revenue'].rolling(window=7, min_periods=1).mean()
            fig.add_trace(go.Scatter(
                x=revenue_trend['date'],
                y=revenue_trend['ma'],
                mode='lines',
                name='7-Period MA',
                line=dict(color=COLORS['secondary'], width=2, dash='dash'),
                hovertemplate='<b>%{x|%Y-%m-%d}</b><br>MA: ₹%{y:,.0f}<extra></extra>'
            ))
        
        fig.update_layout(
            title=f'{aggregation} Revenue Trend',
            xaxis_title='Date',
            yaxis_title='Revenue (₹)',
            hovermode='x unified',
            height=400,
            template='plotly_white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Channel Performance Comparison
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Revenue by Channel")
        
        channel_metrics = campaigns.groupby('channel').agg({
            'revenue': 'sum',
            'spend': 'sum',
            'conversions': 'sum'
        }).reset_index()
        channel_metrics['roi'] = (channel_metrics['revenue'] - channel_metrics['spend']) / channel_metrics['spend'] * 100
        channel_metrics = channel_metrics.sort_values('revenue', ascending=True)
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            y=channel_metrics['channel'],
            x=channel_metrics['revenue'],
            orientation='h',
            marker=dict(
                color=channel_metrics['roi'],
                colorscale='RdYlGn',
                showscale=True,
                colorbar=dict(title="ROI %")
            ),
            text=channel_metrics['revenue'].apply(lambda x: f'₹{x/1e6:.1f}M'),
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>Revenue: ₹%{x:,.0f}<br>ROI: %{marker.color:.1f}%<extra></extra>'
        ))
        
        fig.update_layout(
            title='Channel Revenue (Color: ROI %)',
            xaxis_title='Revenue (₹)',
            yaxis_title='',
            height=400,
            template='plotly_white',
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Campaign Type Performance")
        
        campaign_type_perf = campaigns.groupby('campaign_type').agg({
            'revenue': 'sum',
            'conversions': 'sum',
            'spend': 'sum'
        }).reset_index()
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=campaign_type_perf['campaign_type'],
            y=campaign_type_perf['revenue'],
            name='Revenue',
            marker_color=COLORS['primary'],
            text=campaign_type_perf['revenue'].apply(lambda x: f'₹{x/1e6:.1f}M'),
            textposition='outside'
        ))
        
        fig.add_trace(go.Bar(
            x=campaign_type_perf['campaign_type'],
            y=campaign_type_perf['spend'],
            name='Spend',
            marker_color=COLORS['danger'],
            text=campaign_type_perf['spend'].apply(lambda x: f'₹{x/1e6:.1f}M'),
            textposition='inside'
        ))
        
        fig.update_layout(
            title='Revenue vs Spend by Campaign Type',
            xaxis_title='Campaign Type',
            yaxis_title='Amount (₹)',
            barmode='group',
            height=400,
            template='plotly_white',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Regional Performance
    st.subheader("🌍 Regional Performance Overview")
    
    regional_perf = campaigns.groupby('region').agg({
        'revenue': 'sum',
        'conversions': 'sum',
        'impressions': 'sum',
        'clicks': 'sum'
    }).reset_index()
    regional_perf['ctr'] = (regional_perf['clicks'] / regional_perf['impressions'] * 100).round(2)
    regional_perf['conversion_rate'] = (regional_perf['conversions'] / regional_perf['clicks'] * 100).round(2)
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Revenue by Region', 'CTR vs Conversion Rate'),
        specs=[[{"type": "bar"}, {"type": "scatter"}]]
    )
    
    fig.add_trace(
        go.Bar(
            x=regional_perf['region'],
            y=regional_perf['revenue'],
            marker_color=SEQUENTIAL_COLORS[:len(regional_perf)],
            text=regional_perf['revenue'].apply(lambda x: f'₹{x/1e6:.1f}M'),
            textposition='outside',
            showlegend=False
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=regional_perf['ctr'],
            y=regional_perf['conversion_rate'],
            mode='markers+text',
            marker=dict(size=regional_perf['revenue']/1e6, color=COLORS['primary'], opacity=0.6, sizemode='diameter'),
            text=regional_perf['region'],
            textposition='top center',
            showlegend=False,
            hovertemplate='<b>%{text}</b><br>CTR: %{x:.2f}%<br>Conv Rate: %{y:.2f}%<extra></extra>'
        ),
        row=1, col=2
    )
    
    fig.update_xaxes(title_text="Region", row=1, col=1)
    fig.update_yaxes(title_text="Revenue (₹)", row=1, col=1)
    fig.update_xaxes(title_text="CTR (%)", row=1, col=2)
    fig.update_yaxes(title_text="Conversion Rate (%)", row=1, col=2)
    
    fig.update_layout(height=400, template='plotly_white')
    
    st.plotly_chart(fig, use_container_width=True)

# [KEEP ALL OTHER PAGE FUNCTIONS FROM YOUR ORIGINAL CODE]
# I'm not including them here to keep the response concise, but you should keep:
# - page_campaign_analytics
# - page_customer_insights
# - page_product_performance
# - page_geographic_analysis
# - page_attribution_funnel
# - page_customer_journey
# - page_ml_evaluation

# =============================================================================
# MAIN APP
# =============================================================================
def main():
    """Main application controller with flexible data loading"""
    
    # Try to load data automatically
    data = load_data()
    
    # If no data found, show upload interface
    if data is None:
        data = show_file_uploader()
        
        if data is None:
            st.info("""
            ### 💡 Tips:
            - Place all CSV files in the same directory as this script, OR
            - Place them in a 'data/' or 'marketing_dataset/' folder, OR
            - Upload them using the file uploader above
            """)
            st.stop()
    
    # Show success message
    st.sidebar.success("✅ Data loaded successfully!")
    
    # Sidebar navigation
    page = sidebar()
    
    # Route to pages
    page_map = {
        "🏠 Executive Overview": page_executive_overview,
        # Add all other pages here
        # "📈 Campaign Analytics": page_campaign_analytics,
        # etc.
    }
    
    # Display selected page
    if page in page_map:
        page_map[page](data)
    else:
        st.info(f"Page '{page}' is not yet implemented in this shortened version. Please add the function.")

if __name__ == "__main__":
    main()
