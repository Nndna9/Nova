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

# =============================================================================
# PAGE: CAMPAIGN ANALYTICS
# =============================================================================
def page_campaign_analytics(data):
    """Advanced Campaign Analytics with Interactive Filters"""
    st.title("📈 Campaign Analytics")
    st.markdown("### Deep Dive into Campaign Performance")
    
    campaigns = data['campaigns']
    
    # Advanced Filters
    st.markdown("#### 🔍 Filter Campaigns")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        selected_channels = st.multiselect(
            "Channels",
            options=sorted(campaigns['channel'].unique()),
            default=sorted(campaigns['channel'].unique())
        )
    
    with col2:
        selected_regions = st.multiselect(
            "Regions",
            options=sorted(campaigns['region'].unique()),
            default=sorted(campaigns['region'].unique())
        )
    
    with col3:
        selected_types = st.multiselect(
            "Campaign Types",
            options=sorted(campaigns['campaign_type'].unique()),
            default=sorted(campaigns['campaign_type'].unique())
        )
    
    with col4:
        date_range = st.date_input(
            "Date Range",
            value=(campaigns['date'].min(), campaigns['date'].max()),
            min_value=campaigns['date'].min(),
            max_value=campaigns['date'].max()
        )
    
    # Apply filters
    filtered = campaigns[
        (campaigns['channel'].isin(selected_channels)) &
        (campaigns['region'].isin(selected_regions)) &
        (campaigns['campaign_type'].isin(selected_types)) &
        (campaigns['date'] >= pd.to_datetime(date_range[0])) &
        (campaigns['date'] <= pd.to_datetime(date_range[1]))
    ]
    
    # Summary metrics for filtered data
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Campaigns", f"{filtered['campaign_id'].nunique():,}")
    with col2:
        st.metric("Total Spend", f"₹{filtered['spend'].sum()/1e6:.2f}M")
    with col3:
        st.metric("Total Revenue", f"₹{filtered['revenue'].sum()/1e6:.2f}M")
    with col4:
        st.metric("Avg ROAS", f"{filtered['roas'].mean():.2f}x")
    
    st.markdown("---")
    
    # Regional Performance by Quarter
    st.subheader("📊 Regional Performance by Quarter")
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        metric_choice = st.selectbox(
            "Select Metric",
            ["revenue", "conversions", "spend", "roas"],
            format_func=lambda x: x.replace('_', ' ').title()
        )
    
    with col1:
        regional_quarterly = filtered.groupby(['region', 'quarter'])[metric_choice].sum().reset_index()
        
        fig = px.bar(
            regional_quarterly,
            x='quarter',
            y=metric_choice,
            color='region',
            barmode='group',
            title=f'{metric_choice.replace("_", " ").title()} by Region and Quarter',
            labels={metric_choice: metric_choice.replace('_', ' ').title(), 'quarter': 'Quarter', 'region': 'Region'},
            color_discrete_sequence=SEQUENTIAL_COLORS,
            text_auto='.2s'
        )
        
        fig.update_layout(
            height=450,
            template='plotly_white',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Channel Contribution Over Time
    st.subheader("📈 Channel Contribution Over Time")
    
    tabs = st.tabs(["Stacked Area", "Line Chart", "Percentage View"])
    
    with tabs[0]:
        channel_time = filtered.groupby([pd.Grouper(key='date', freq='W'), 'channel'])['conversions'].sum().reset_index()
        
        fig = px.area(
            channel_time,
            x='date',
            y='conversions',
            color='channel',
            title='Weekly Conversions by Channel (Stacked)',
            labels={'date': 'Week', 'conversions': 'Conversions', 'channel': 'Channel'},
            color_discrete_sequence=SEQUENTIAL_COLORS
        )
        
        fig.update_layout(
            height=450,
            template='plotly_white',
            hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tabs[1]:
        fig = px.line(
            channel_time,
            x='date',
            y='conversions',
            color='channel',
            title='Weekly Conversions by Channel (Lines)',
            labels={'date': 'Week', 'conversions': 'Conversions', 'channel': 'Channel'},
            color_discrete_sequence=SEQUENTIAL_COLORS,
            markers=True
        )
        
        fig.update_layout(
            height=450,
            template='plotly_white',
            hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tabs[2]:
        channel_time_pct = channel_time.pivot_table(
            index='date', 
            columns='channel', 
            values='conversions', 
            fill_value=0
        )
        channel_time_pct = channel_time_pct.div(channel_time_pct.sum(axis=1), axis=0) * 100
        channel_time_pct = channel_time_pct.reset_index().melt(id_vars='date', var_name='channel', value_name='percentage')
        
        fig = px.area(
            channel_time_pct,
            x='date',
            y='percentage',
            color='channel',
            title='Channel Contribution % Over Time',
            labels={'date': 'Week', 'percentage': 'Percentage (%)', 'channel': 'Channel'},
            color_discrete_sequence=SEQUENTIAL_COLORS
        )
        
        fig.update_layout(
            height=450,
            template='plotly_white',
            hovermode='x unified',
            yaxis=dict(range=[0, 100]),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Campaign Performance Heatmap
    st.subheader("🔥 Campaign Performance Heatmap")
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        heatmap_metric = st.selectbox(
            "Heatmap Metric",
            ["revenue", "conversions", "roas", "ctr"],
            format_func=lambda x: x.replace('_', ' ').title()
        )
    
    with col1:
        # Create day of week and week number
        filtered['day_of_week'] = filtered['date'].dt.day_name()
        filtered['week_num'] = filtered['date'].dt.isocalendar().week
        
        heatmap_data = filtered.groupby(['day_of_week', 'week_num'])[heatmap_metric].mean().reset_index()
        heatmap_pivot = heatmap_data.pivot(index='day_of_week', columns='week_num', values=heatmap_metric)
        
        # Reorder days
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        heatmap_pivot = heatmap_pivot.reindex([d for d in day_order if d in heatmap_pivot.index])
        
        fig = px.imshow(
            heatmap_pivot,
            labels=dict(x="Week Number", y="Day of Week", color=heatmap_metric.title()),
            x=heatmap_pivot.columns,
            y=heatmap_pivot.index,
            color_continuous_scale='Blues',
            aspect='auto',
            title=f'Average {heatmap_metric.title()} by Day and Week'
        )
        
        fig.update_layout(
            height=400,
            template='plotly_white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Top Performing Campaigns
    st.subheader("🏆 Top Performing Campaigns")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        top_n = st.slider("Show Top N Campaigns", 5, 20, 10)
        sort_by = st.selectbox(
            "Sort By",
            ["revenue", "conversions", "roas"],
            format_func=lambda x: x.replace('_', ' ').title()
        )
    
    with col2:
        top_campaigns = filtered.groupby('campaign_name').agg({
            'revenue': 'sum',
            'spend': 'sum',
            'conversions': 'sum',
            'roas': 'mean'
        }).reset_index().sort_values(sort_by, ascending=False).head(top_n)
        
        st.dataframe(
            top_campaigns.style.format({
                'revenue': '₹{:,.0f}',
                'spend': '₹{:,.0f}',
                'conversions': '{:,.0f}',
                'roas': '{:.2f}x'
            }).background_gradient(subset=[sort_by], cmap='Blues'),
            use_container_width=True,
            height=400
        )

# =============================================================================
# PAGE: CUSTOMER INSIGHTS
# =============================================================================
def page_customer_insights(data):
    """Customer Insights and Segmentation Analysis"""
    st.title("👥 Customer Insights")
    st.markdown("### Understanding Customer Behavior and Segments")
    
    customers = data['customers']
    
    # Customer Overview Metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Customers", f"{len(customers):,}")
    with col2:
        avg_ltv = customers['lifetime_value'].mean()
        st.metric("Avg LTV", f"₹{avg_ltv:,.0f}")
    with col3:
        avg_satisfaction = customers['satisfaction_score'].mean()
        st.metric("Avg Satisfaction", f"{avg_satisfaction:.2f}/5")
    with col4:
        churn_rate = customers['is_churned'].mean() * 100
        st.metric("Churn Rate", f"{churn_rate:.1f}%")
    with col5:
        promoters = (customers['nps_category'] == 'Promoter').sum()
        st.metric("Promoters", f"{promoters:,}")
    
    st.markdown("---")
    
    # Demographics Analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Age Distribution")
        
        bin_size = st.slider("Bin Size (years)", min_value=2, max_value=10, value=5, key='age_bin')
        
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=customers['age'],
            nbinsx=int((customers['age'].max() - customers['age'].min()) / bin_size),
            marker_color=COLORS['primary'],
            opacity=0.75,
            name='Age Distribution',
            hovertemplate='Age Range: %{x}<br>Count: %{y}<extra></extra>'
        ))
        
        # Add mean line
        mean_age = customers['age'].mean()
        fig.add_vline(
            x=mean_age, 
            line_dash="dash", 
            line_color=COLORS['danger'],
            annotation_text=f"Mean: {mean_age:.1f}",
            annotation_position="top"
        )
        
        fig.update_layout(
            title='Customer Age Distribution',
            xaxis_title='Age',
            yaxis_title='Count',
            height=400,
            template='plotly_white',
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("👫 Gender & Age Group Distribution")
        
        gender_age = customers.groupby(['gender', 'age_group']).size().reset_index(name='count')
        
        fig = px.bar(
            gender_age,
            x='age_group',
            y='count',
            color='gender',
            barmode='group',
            title='Customer Distribution by Gender and Age Group',
            labels={'count': 'Count', 'age_group': 'Age Group', 'gender': 'Gender'},
            color_discrete_sequence=[COLORS['primary'], COLORS['secondary'], COLORS['success']],
            text_auto=True
        )
        
        fig.update_layout(
            height=400,
            template='plotly_white',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Lifetime Value Analysis
    st.subheader("💰 Lifetime Value Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### LTV by Customer Segment")
        
        fig = go.Figure()
        
        for segment in customers['customer_segment'].unique():
            segment_data = customers[customers['customer_segment'] == segment]['lifetime_value']
            
            fig.add_trace(go.Box(
                y=segment_data,
                name=segment,
                boxmean='sd',
                marker_color=SEQUENTIAL_COLORS[list(customers['customer_segment'].unique()).index(segment) % len(SEQUENTIAL_COLORS)],
                hovertemplate='<b>%{fullData.name}</b><br>LTV: ₹%{y:,.0f}<extra></extra>'
            ))
        
        fig.update_layout(
            title='LTV Distribution by Customer Segment',
            yaxis_title='Lifetime Value (₹)',
            height=450,
            template='plotly_white',
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("##### LTV by Income Bracket")
        
        fig = go.Figure()
        
        for bracket in sorted(customers['income_bracket'].unique()):
            bracket_data = customers[customers['income_bracket'] == bracket]['lifetime_value']
            
            fig.add_trace(go.Violin(
                y=bracket_data,
                name=bracket,
                box_visible=True,
                meanline_visible=True,
                fillcolor=SEQUENTIAL_COLORS[list(sorted(customers['income_bracket'].unique())).index(bracket) % len(SEQUENTIAL_COLORS)],
                opacity=0.6,
                hovertemplate='<b>%{fullData.name}</b><br>LTV: ₹%{y:,.0f}<extra></extra>'
            ))
        
        fig.update_layout(
            title='LTV Distribution by Income Bracket',
            yaxis_title='Lifetime Value (₹)',
            height=450,
            template='plotly_white',
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Income vs LTV Scatter
    st.subheader("🔵 Income vs Lifetime Value Analysis")
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        color_by = st.selectbox(
            "Color By",
            ["customer_segment", "nps_category", "region", "acquisition_channel"]
        )
        
        show_trendline = st.checkbox("Show Trendline", value=True)
    
    with col1:
        fig = px.scatter(
            customers,
            x='income',
            y='lifetime_value',
            color=color_by,
            size='total_purchases',
            hover_data=['age', 'tenure_months', 'satisfaction_score', 'avg_order_value'],
            title=f'Income vs Lifetime Value (colored by {color_by.replace("_", " ").title()})',
            labels={
                'income': 'Annual Income (₹)',
                'lifetime_value': 'Lifetime Value (₹)',
                color_by: color_by.replace('_', ' ').title()
            },
            color_discrete_sequence=SEQUENTIAL_COLORS,
            trendline='ols' if show_trendline else None
        )
        
        fig.update_layout(
            height=500,
            template='plotly_white',
            legend=dict(orientation="v", yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Customer Satisfaction Analysis
    st.subheader("😊 Customer Satisfaction & NPS Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### Satisfaction Score by NPS Category")
        
        fig = go.Figure()
        
        for category in ['Detractor', 'Passive', 'Promoter']:
            if category in customers['nps_category'].values:
                category_data = customers[customers['nps_category'] == category]['satisfaction_score']
                
                fig.add_trace(go.Violin(
                    y=category_data,
                    name=category,
                    box_visible=True,
                    meanline_visible=True,
                    fillcolor={'Detractor': COLORS['danger'], 'Passive': COLORS['warning'], 'Promoter': COLORS['success']}[category],
                    opacity=0.6,
                    hovertemplate='<b>%{fullData.name}</b><br>Score: %{y:.2f}<extra></extra>'
                ))
        
        fig.update_layout(
            title='Satisfaction Score Distribution by NPS Category',
            yaxis_title='Satisfaction Score',
            height=400,
            template='plotly_white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("##### NPS Category Distribution")
        
        nps_dist = customers['nps_category'].value_counts().reset_index()
        nps_dist.columns = ['category', 'count']
        
        fig = go.Figure(data=[go.Pie(
            labels=nps_dist['category'],
            values=nps_dist['count'],
            hole=0.4,
            marker=dict(colors=[COLORS['danger'], COLORS['warning'], COLORS['success']]),
            textinfo='label+percent',
            hovertemplate='<b>%{label}</b><br>Count: %{value:,}<br>Percentage: %{percent}<extra></extra>'
        )])
        
        fig.update_layout(
            title='NPS Category Distribution',
            height=400,
            template='plotly_white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Churn Analysis
    st.subheader("⚠️ Churn Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### Churn Rate by Segment")
        
        churn_by_segment = customers.groupby('customer_segment').agg({
            'is_churned': 'mean',
            'customer_id': 'count'
        }).reset_index()
        churn_by_segment.columns = ['segment', 'churn_rate', 'total_customers']
        churn_by_segment['churn_rate'] = churn_by_segment['churn_rate'] * 100
        churn_by_segment = churn_by_segment.sort_values('churn_rate', ascending=True)
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            y=churn_by_segment['segment'],
            x=churn_by_segment['churn_rate'],
            orientation='h',
            marker=dict(
                color=churn_by_segment['churn_rate'],
                colorscale='Reds',
                showscale=True,
                colorbar=dict(title="Churn %")
            ),
            text=churn_by_segment['churn_rate'].apply(lambda x: f'{x:.1f}%'),
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>Churn Rate: %{x:.1f}%<br>Total: %{customdata:,}<extra></extra>',
            customdata=churn_by_segment['total_customers']
        ))
        
        fig.update_layout(
            title='Churn Rate by Customer Segment',
            xaxis_title='Churn Rate (%)',
            yaxis_title='',
            height=400,
            template='plotly_white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("##### Churn Probability Distribution")
        
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=customers[customers['is_churned'] == 0]['churn_probability'],
            name='Active',
            marker_color=COLORS['success'],
            opacity=0.7,
            nbinsx=30
        ))
        
        fig.add_trace(go.Histogram(
            x=customers[customers['is_churned'] == 1]['churn_probability'],
            name='Churned',
            marker_color=COLORS['danger'],
            opacity=0.7,
            nbinsx=30
        ))
        
        fig.update_layout(
            title='Churn Probability Distribution',
            xaxis_title='Churn Probability',
            yaxis_title='Count',
            barmode='overlay',
            height=400,
            template='plotly_white',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Customer Segmentation Summary Table
    st.subheader("📋 Customer Segment Summary")
    
    segment_summary = customers.groupby('customer_segment').agg({
        'customer_id': 'count',
        'lifetime_value': 'mean',
        'total_purchases': 'mean',
        'avg_order_value': 'mean',
        'satisfaction_score': 'mean',
        'is_churned': lambda x: (x.mean() * 100),
        'tenure_months': 'mean'
    }).reset_index()
    
    segment_summary.columns = ['Segment', 'Customers', 'Avg LTV', 'Avg Purchases', 'Avg Order Value', 'Avg Satisfaction', 'Churn Rate %', 'Avg Tenure (months)']
    
    st.dataframe(
        segment_summary.style.format({
            'Customers': '{:,.0f}',
            'Avg LTV': '₹{:,.0f}',
            'Avg Purchases': '{:.1f}',
            'Avg Order Value': '₹{:,.0f}',
            'Avg Satisfaction': '{:.2f}',
            'Churn Rate %': '{:.1f}%',
            'Avg Tenure (months)': '{:.1f}'
        }).background_gradient(subset=['Avg LTV'], cmap='Greens')
        .background_gradient(subset=['Churn Rate %'], cmap='Reds'),
        use_container_width=True
    )

# =============================================================================
# PAGE: PRODUCT PERFORMANCE
# =============================================================================
def page_product_performance(data):
    """Product Performance and Category Analysis"""
    st.title("📦 Product Performance")
    st.markdown("### Product Sales and Profitability Analysis")
    
    products = data['products']
    
    # Product Metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        total_sales = products['sales'].sum()
        st.metric("Total Sales", f"₹{total_sales/1e7:.2f} Cr")
    with col2:
        total_profit = products['profit'].sum()
        st.metric("Total Profit", f"₹{total_profit/1e6:.2f}M")
    with col3:
        avg_margin = products['profit_margin'].mean()
        st.metric("Avg Margin", f"{avg_margin:.1f}%")
    with col4:
        avg_rating = products['avg_rating'].mean()
        st.metric("Avg Rating", f"{avg_rating:.2f}/5")
    with col5:
        avg_return = products['return_rate'].mean()
        st.metric("Avg Return Rate", f"{avg_return:.1f}%")
    
    st.markdown("---")
    
    # Product Hierarchy Treemap
    st.subheader("🌳 Product Sales Hierarchy")
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        treemap_size = st.selectbox(
            "Size By",
            ["sales", "profit", "units_sold"],
            format_func=lambda x: x.replace('_', ' ').title()
        )
        
        treemap_color = st.selectbox(
            "Color By",
            ["profit_margin", "avg_rating", "return_rate"],
            format_func=lambda x: x.replace('_', ' ').title()
        )
    
    with col1:
        fig = px.treemap(
            products,
            path=['category', 'subcategory', 'product_name'],
            values=treemap_size,
            color=treemap_color,
            color_continuous_scale='RdYlGn' if treemap_color != 'return_rate' else 'RdYlGn_r',
            title=f'Product Hierarchy (Size: {treemap_size.title()}, Color: {treemap_color.replace("_", " ").title()})',
            hover_data=['sales', 'profit', 'profit_margin', 'avg_rating']
        )
        
        fig.update_layout(
            height=600,
            template='plotly_white'
        )
        
        fig.update_traces(
            textinfo='label+value',
            hovertemplate='<b>%{label}</b><br>Sales: ₹%{customdata[0]:,.0f}<br>Profit: ₹%{customdata[1]:,.0f}<br>Margin: %{customdata[2]:.1f}%<br>Rating: %{customdata[3]:.2f}<extra></extra>'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Sunburst Chart
    st.subheader("☀️ Sales Distribution: Category → Region")
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        sunburst_metric = st.selectbox(
            "Metric",
            ["sales", "profit", "units_sold"],
            format_func=lambda x: x.replace('_', ' ').title(),
            key='sunburst_metric'
        )
    
    with col1:
        category_region = products.groupby(['category', 'region'])[sunburst_metric].sum().reset_index()
        
        fig = px.sunburst(
            category_region,
            path=['category', 'region'],
            values=sunburst_metric,
            title=f'{sunburst_metric.replace("_", " ").title()} Distribution: Category → Region',
            color=sunburst_metric,
            color_continuous_scale='Blues'
        )
        
        fig.update_layout(
            height=600,
            template='plotly_white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Category Performance Comparison
    st.subheader("📊 Category Performance Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### Sales vs Profit by Category")
        
        category_perf = products.groupby('category').agg({
            'sales': 'sum',
            'profit': 'sum',
            'profit_margin': 'mean',
            'units_sold': 'sum'
        }).reset_index()
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=category_perf['sales'],
            y=category_perf['profit'],
            mode='markers+text',
            marker=dict(
                size=category_perf['units_sold']/1000,
                color=category_perf['profit_margin'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Margin %"),
                line=dict(width=1, color='white')
            ),
            text=category_perf['category'],
            textposition='top center',
            hovertemplate='<b>%{text}</b><br>Sales: ₹%{x:,.0f}<br>Profit: ₹%{y:,.0f}<br>Margin: %{marker.color:.1f}%<extra></extra>'
        ))
        
        fig.update_layout(
            title='Sales vs Profit (Size: Units Sold, Color: Margin %)',
            xaxis_title='Sales (₹)',
            yaxis_title='Profit (₹)',
            height=450,
            template='plotly_white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("##### Profit Margin by Category")
        
        category_margin = products.groupby('category')['profit_margin'].mean().sort_values(ascending=True).reset_index()
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            y=category_margin['category'],
            x=category_margin['profit_margin'],
            orientation='h',
            marker=dict(
                color=category_margin['profit_margin'],
                colorscale='RdYlGn',
                showscale=False
            ),
            text=category_margin['profit_margin'].apply(lambda x: f'{x:.1f}%'),
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>Margin: %{x:.1f}%<extra></extra>'
        ))
        
        fig.update_layout(
            title='Average Profit Margin by Category',
            xaxis_title='Profit Margin (%)',
            yaxis_title='',
            height=450,
            template='plotly_white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Product Quality Metrics
    st.subheader("⭐ Product Quality Metrics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### Rating vs Return Rate")
        
        fig = px.scatter(
            products,
            x='avg_rating',
            y='return_rate',
            size='sales',
            color='category',
            hover_data=['product_name', 'sales', 'profit_margin'],
            title='Product Rating vs Return Rate (Size: Sales)',
            labels={'avg_rating': 'Average Rating', 'return_rate': 'Return Rate (%)'},
            color_discrete_sequence=SEQUENTIAL_COLORS
        )
        
        fig.update_layout(
            height=450,
            template='plotly_white',
            legend=dict(orientation="v", yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("##### Review Count Distribution")
        
        fig = go.Figure()
        
        for category in products['category'].unique():
            category_data = products[products['category'] == category]['review_count']
            
            fig.add_trace(go.Box(
                y=category_data,
                name=category,
                boxmean='sd',
                marker_color=SEQUENTIAL_COLORS[list(products['category'].unique()).index(category) % len(SEQUENTIAL_COLORS)]
            ))
        
        fig.update_layout(
            title='Review Count Distribution by Category',
            yaxis_title='Number of Reviews',
            height=450,
            template='plotly_white',
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Regional Product Performance
    st.subheader("🌍 Regional Product Performance")
    
    regional_product = products.groupby(['region', 'category']).agg({
        'sales': 'sum',
        'profit': 'sum',
        'units_sold': 'sum'
    }).reset_index()
    
    fig = px.bar(
        regional_product,
        x='region',
        y='sales',
        color='category',
        title='Sales by Region and Category',
        labels={'sales': 'Sales (₹)', 'region': 'Region', 'category': 'Category'},
        color_discrete_sequence=SEQUENTIAL_COLORS,
        barmode='stack',
        text_auto='.2s'
    )
    
    fig.update_layout(
        height=450,
        template='plotly_white',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Top Products Table
    st.subheader("🏆 Top Performing Products")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        top_n_products = st.slider("Show Top N Products", 5, 20, 10, key='top_products')
        sort_by_product = st.selectbox(
            "Sort By",
            ["sales", "profit", "profit_margin", "avg_rating"],
            format_func=lambda x: x.replace('_', ' ').title(),
            key='sort_products'
        )
    
    with col2:
        top_products = products.nlargest(top_n_products, sort_by_product)[
            ['product_name', 'category', 'subcategory', 'sales', 'profit', 'profit_margin', 'avg_rating', 'return_rate']
        ]
        
        st.dataframe(
            top_products.style.format({
                'sales': '₹{:,.0f}',
                'profit': '₹{:,.0f}',
                'profit_margin': '{:.1f}%',
                'avg_rating': '{:.2f}',
                'return_rate': '{:.1f}%'
            }).background_gradient(subset=[sort_by_product], cmap='Greens'),
            use_container_width=True,
            height=400
        )

# =============================================================================
# PAGE: GEOGRAPHIC ANALYSIS
# =============================================================================
def page_geographic_analysis(data):
    """Geographic Performance Analysis"""
    st.title("🗺️ Geographic Analysis")
    st.markdown("### State-wise Performance Metrics")
    
    geo = data['geographic']
    
    # Geographic Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_states = len(geo)
        st.metric("States Covered", f"{total_states}")
    with col2:
        total_stores = geo['store_count'].sum()
        st.metric("Total Stores", f"{total_stores:,}")
    with col3:
        avg_penetration = geo['market_penetration'].mean()
        st.metric("Avg Market Penetration", f"{avg_penetration:.1f}%")
    with col4:
        avg_growth = geo['yoy_growth'].mean()
        st.metric("Avg YoY Growth", f"{avg_growth:.1f}%")
    
    st.markdown("---")
    
    # Bubble Map
    st.subheader("📍 State-wise Performance Map")
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        metric = st.selectbox(
            "Select Metric",
            ['total_revenue', 'total_customers', 'market_penetration', 'yoy_growth', 'customer_satisfaction'],
            format_func=lambda x: x.replace('_', ' ').title()
        )
        
        color_metric = st.selectbox(
            "Color By",
            ['region', 'yoy_growth', 'customer_satisfaction', 'market_penetration'],
            format_func=lambda x: x.replace('_', ' ').title()
        )
    
    with col1:
        fig = px.scatter_geo(
            geo,
            lat='latitude',
            lon='longitude',
            size=metric,
            color=color_metric if color_metric != 'region' else 'region',
            hover_name='state',
            hover_data={
                'total_revenue': ':,.0f',
                'total_customers': ':,',
                'store_count': ':,',
                'market_penetration': ':.1f',
                'yoy_growth': ':.1f',
                'customer_satisfaction': ':.2f',
                'latitude': False,
                'longitude': False
            },
            title=f'State Performance - {metric.replace("_", " ").title()} (Color: {color_metric.replace("_", " ").title()})',
            scope='asia',
            color_continuous_scale='Viridis' if color_metric != 'region' else None,
            color_discrete_sequence=SEQUENTIAL_COLORS if color_metric == 'region' else None
        )
        
        fig.update_geos(
            center=dict(lat=20.5937, lon=78.9629),
            projection_scale=4,
            showland=True,
            landcolor='#f0f0f0',
            showlakes=True,
            lakecolor='#e6f2ff',
            showcountries=True,
            countrycolor='#cccccc'
        )
        
        fig.update_layout(
            height=600,
            template='plotly_white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Regional Comparison
    st.subheader("📊 Regional Performance Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### Revenue & Customers by Region")
        
        regional_summary = geo.groupby('region').agg({
            'total_revenue': 'sum',
            'total_customers': 'sum',
            'store_count': 'sum'
        }).reset_index()
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig.add_trace(
            go.Bar(
                x=regional_summary['region'],
                y=regional_summary['total_revenue'],
                name='Revenue',
                marker_color=COLORS['primary'],
                text=regional_summary['total_revenue'].apply(lambda x: f'₹{x/1e6:.1f}M'),
                textposition='outside'
            ),
            secondary_y=False
        )
        
        fig.add_trace(
            go.Scatter(
                x=regional_summary['region'],
                y=regional_summary['total_customers'],
                name='Customers',
                mode='lines+markers',
                marker=dict(size=10, color=COLORS['secondary']),
                line=dict(width=3)
            ),
            secondary_y=True
        )
        
        fig.update_xaxes(title_text="Region")
        fig.update_yaxes(title_text="Revenue (₹)", secondary_y=False)
        fig.update_yaxes(title_text="Customers", secondary_y=True)
        
        fig.update_layout(
            title='Revenue & Customers by Region',
            height=450,
            template='plotly_white',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("##### Market Penetration vs YoY Growth")
        
        fig = px.scatter(
            geo,
            x='market_penetration',
            y='yoy_growth',
            size='total_revenue',
            color='region',
            hover_name='state',
            hover_data=['total_customers', 'store_count', 'customer_satisfaction'],
            title='Market Penetration vs Growth (Size: Revenue)',
            labels={
                'market_penetration': 'Market Penetration (%)',
                'yoy_growth': 'YoY Growth (%)'
            },
            color_discrete_sequence=SEQUENTIAL_COLORS
        )
        
        # Add quadrant lines
        avg_penetration = geo['market_penetration'].mean()
        avg_growth = geo['yoy_growth'].mean()
        
        fig.add_hline(y=avg_growth, line_dash="dash", line_color="gray", opacity=0.5)
        fig.add_vline(x=avg_penetration, line_dash="dash", line_color="gray", opacity=0.5)
        
        fig.update_layout(
            height=450,
            template='plotly_white',
            legend=dict(orientation="v", yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # State Performance Metrics
    st.subheader("📈 State Performance Heatmap")
    
    # Normalize metrics for heatmap
    heatmap_data = geo[['state', 'total_revenue', 'total_customers', 'market_penetration', 'yoy_growth', 'customer_satisfaction']].copy()
    
    # Normalize to 0-100 scale
    for col in ['total_revenue', 'total_customers', 'market_penetration', 'yoy_growth', 'customer_satisfaction']:
        heatmap_data[f'{col}_norm'] = (heatmap_data[col] - heatmap_data[col].min()) / (heatmap_data[col].max() - heatmap_data[col].min()) * 100
    
    heatmap_matrix = heatmap_data[['state', 'total_revenue_norm', 'total_customers_norm', 'market_penetration_norm', 'yoy_growth_norm', 'customer_satisfaction_norm']].set_index('state')
    heatmap_matrix.columns = ['Revenue', 'Customers', 'Penetration', 'Growth', 'Satisfaction']
    
    fig = px.imshow(
        heatmap_matrix.T,
        labels=dict(x="State", y="Metric", color="Score (0-100)"),
        x=heatmap_matrix.index,
        y=heatmap_matrix.columns,
        color_continuous_scale='RdYlGn',
        aspect='auto',
        title='State Performance Heatmap (Normalized Scores)'
    )
    
    fig.update_layout(
        height=400,
        template='plotly_white'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed State Metrics Table
    st.subheader("📋 Detailed State Metrics")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        sort_column = st.selectbox(
            "Sort By",
            ['total_revenue', 'total_customers', 'market_penetration', 'yoy_growth', 'customer_satisfaction'],
            format_func=lambda x: x.replace('_', ' ').title()
        )
        
        ascending = st.checkbox("Ascending Order", value=False)
    
    with col2:
        sorted_geo = geo.sort_values(sort_column, ascending=ascending)
        
        st.dataframe(
            sorted_geo[['state', 'region', 'total_customers', 'total_revenue', 'revenue_per_customer', 
                        'store_count', 'market_penetration', 'yoy_growth', 'customer_satisfaction', 'avg_delivery_days']]
            .style.format({
                'total_customers': '{:,}',
                'total_revenue': '₹{:,.0f}',
                'revenue_per_customer': '₹{:,.0f}',
                'store_count': '{:,}',
                'market_penetration': '{:.1f}%',
                'yoy_growth': '{:.1f}%',
                'customer_satisfaction': '{:.2f}',
                'avg_delivery_days': '{:.1f}'
            }).background_gradient(subset=[sort_column], cmap='Greens'),
            use_container_width=True,
            height=400
        )
    
    # Regional Summary
    st.subheader("🌍 Regional Summary")
    
    regional_summary_full = geo.groupby('region').agg({
        'state': 'count',
        'total_customers': 'sum',
        'total_revenue': 'sum',
        'store_count': 'sum',
        'market_penetration': 'mean',
        'yoy_growth': 'mean',
        'customer_satisfaction': 'mean',
        'avg_delivery_days': 'mean'
    }).reset_index()
    
    regional_summary_full.columns = ['Region', 'States', 'Total Customers', 'Total Revenue', 'Stores', 
                                     'Avg Penetration %', 'Avg Growth %', 'Avg Satisfaction', 'Avg Delivery Days']
    
    st.dataframe(
        regional_summary_full.style.format({
            'States': '{:,}',
            'Total Customers': '{:,}',
            'Total Revenue': '₹{:,.0f}',
            'Stores': '{:,}',
            'Avg Penetration %': '{:.1f}%',
            'Avg Growth %': '{:.1f}%',
            'Avg Satisfaction': '{:.2f}',
            'Avg Delivery Days': '{:.1f}'
        }).background_gradient(subset=['Total Revenue'], cmap='Blues')
        .background_gradient(subset=['Avg Satisfaction'], cmap='Greens'),
        use_container_width=True
    )

# =============================================================================
# PAGE: ATTRIBUTION & FUNNEL
# =============================================================================
def page_attribution_funnel(data):
    """Attribution Models and Funnel Analysis"""
    st.title("🎯 Attribution & Funnel Analysis")
    st.markdown("### Marketing Attribution and Conversion Funnel")
    
    attribution = data['attribution']
    funnel = data['funnel']
    correlation = data['correlation']
    
    # Attribution Analysis
    st.subheader("🍩 Channel Attribution Models")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### Attribution Model Comparison")
        
        model = st.selectbox(
            "Select Attribution Model",
            ['first_touch', 'last_touch', 'linear', 'time_decay', 'position_based'],
            format_func=lambda x: x.replace('_', ' ').title()
        )
        
        fig = go.Figure(data=[go.Pie(
            labels=attribution['channel'],
            values=attribution[model],
            hole=0.4,
            marker=dict(colors=SEQUENTIAL_COLORS),
            textinfo='label+percent',
            hovertemplate='<b>%{label}</b><br>Attribution: %{value}%<br>Percentage: %{percent}<extra></extra>'
        )])
        
        fig.update_layout(
            title=f'{model.replace("_", " ").title()} Attribution Model',
            height=450,
            template='plotly_white',
            annotations=[dict(text=model.replace('_', ' ').title(), x=0.5, y=0.5, font_size=16, showarrow=False)]
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("##### Attribution Model Comparison")
        
        # Reshape data for comparison
        attribution_long = attribution.melt(
            id_vars='channel',
            value_vars=['first_touch', 'last_touch', 'linear', 'time_decay', 'position_based'],
            var_name='model',
            value_name='attribution'
        )
        
        fig = px.bar(
            attribution_long,
            x='channel',
            y='attribution',
            color='model',
            barmode='group',
            title='Attribution % by Model and Channel',
            labels={'attribution': 'Attribution %', 'channel': 'Channel', 'model': 'Model'},
            color_discrete_sequence=SEQUENTIAL_COLORS,
            text_auto='.1f'
        )
        
        fig.update_layout(
            height=450,
            template='plotly_white',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Funnel Analysis
    st.subheader("🔻 Marketing Conversion Funnel")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig = go.Figure()
        
        fig.add_trace(go.Funnel(
            y=funnel['stage'],
            x=funnel['visitors'],
            textposition="inside",
            textinfo="value+percent initial+percent previous",
            marker=dict(
                color=SEQUENTIAL_COLORS[:len(funnel)],
                line=dict(width=2, color='white')
            ),
            connector=dict(line=dict(color='gray', width=2)),
            hovertemplate='<b>%{y}</b><br>Visitors: %{x:,}<br>% of Initial: %{percentInitial}<br>% of Previous: %{percentPrevious}<extra></extra>'
        ))
        
        fig.update_layout(
            title='Marketing Funnel - Visitor Flow',
            height=500,
            template='plotly_white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("##### Funnel Metrics")
        
        # Calculate drop-off rates
        funnel['drop_off'] = funnel['visitors'].diff().fillna(0).abs()
        funnel['drop_off_rate'] = (funnel['drop_off'] / funnel['visitors'].shift(1) * 100).fillna(0)
        
        st.dataframe(
            funnel[['stage', 'visitors', 'conversion_rate', 'drop_off_rate']].style.format({
                'visitors': '{:,}',
                'conversion_rate': '{:.1f}%',
                'drop_off_rate': '{:.1f}%'
            }).background_gradient(subset=['conversion_rate'], cmap='Greens')
            .background_gradient(subset=['drop_off_rate'], cmap='Reds'),
            use_container_width=True,
            height=500
        )
    
    # Funnel Conversion Rates
    st.subheader("📊 Stage-wise Conversion Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### Conversion Rate by Stage")
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=funnel['stage'],
            y=funnel['conversion_rate'],
            marker=dict(
                color=funnel['conversion_rate'],
                colorscale='RdYlGn',
                showscale=True,
                colorbar=dict(title="Rate %")
            ),
            text=funnel['conversion_rate'].apply(lambda x: f'{x:.1f}%'),
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Conversion Rate: %{y:.1f}%<extra></extra>'
        ))
        
        fig.update_layout(
            title='Conversion Rate at Each Stage',
            xaxis_title='Funnel Stage',
            yaxis_title='Conversion Rate (%)',
            height=400,
            template='plotly_white',
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("##### Visitor Drop-off")
        
        fig = go.Figure()
        
        fig.add_trace(go.Waterfall(
            x=funnel['stage'],
            y=[-x if i > 0 else x for i, x in enumerate(funnel['visitors'].diff().fillna(funnel['visitors'].iloc[0]))],
            connector={"line": {"color": "gray"}},
            decreasing={"marker": {"color": COLORS['danger']}},
            increasing={"marker": {"color": COLORS['success']}},
            totals={"marker": {"color": COLORS['primary']}},
            text=[f"{x:,.0f}" for x in funnel['visitors']],
            textposition="outside",
            hovertemplate='<b>%{x}</b><br>Visitors: %{text}<extra></extra>'
        ))
        
        fig.update_layout(
            title='Visitor Flow Through Funnel',
            xaxis_title='Funnel Stage',
            yaxis_title='Visitors',
            height=400,
            template='plotly_white',
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Correlation Heatmap
    st.subheader("🔥 Marketing Metrics Correlation")
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        show_values = st.checkbox("Show Values", value=True)
        color_scale = st.selectbox(
            "Color Scale",
            ["RdBu_r", "Viridis", "Cividis", "Blues"],
            index=0
        )
    
    with col1:
        fig = px.imshow(
            correlation,
            text_auto='.2f' if show_values else False,
            aspect='auto',
            color_continuous_scale=color_scale,
            title='Correlation Between Marketing Metrics',
            labels=dict(color="Correlation"),
            zmin=-1,
            zmax=1
        )
        
        fig.update_layout(
            height=600,
            template='plotly_white'
        )
        
        fig.update_xaxes(side="bottom")
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Key Insights
    st.subheader("💡 Key Insights")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info(f"""
        **Funnel Efficiency**
        - Overall Conversion: {funnel['conversion_rate'].iloc[-1]:.1f}%
        - Biggest Drop: {funnel['stage'].iloc[funnel['drop_off_rate'].idxmax()]} ({funnel['drop_off_rate'].max():.1f}%)
        """)
    
    with col2:
        st.success(f"""
        **Attribution Insights**
        - Top First-Touch: {attribution.loc[attribution['first_touch'].idxmax(), 'channel']}
        - Top Last-Touch: {attribution.loc[attribution['last_touch'].idxmax(), 'channel']}
        """)
    
    with col3:
        # Find strongest correlation
        corr_values = correlation.values
        np.fill_diagonal(corr_values, 0)
        max_corr_idx = np.unravel_index(np.argmax(np.abs(corr_values)), corr_values.shape)
        
        st.warning(f"""
        **Correlation Insights**
        - Strongest: {correlation.index[max_corr_idx[0]]} ↔ {correlation.columns[max_corr_idx[1]]}
        - Value: {corr_values[max_corr_idx]:.2f}
        """)

# =============================================================================
# PAGE: CUSTOMER JOURNEY
# =============================================================================
def page_customer_journey(data):
    """Customer Journey and Touchpoint Analysis"""
    st.title("🛤️ Customer Journey Analysis")
    st.markdown("### Multi-Touchpoint Customer Paths")
    
    journey = data['journey']
    
    # Journey Overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_journeys = journey['customer_count'].sum()
        st.metric("Total Journeys", f"{total_journeys:,}")
    with col2:
        unique_paths = len(journey)
        st.metric("Unique Paths", f"{unique_paths:,}")
    with col3:
        avg_customers = journey['customer_count'].mean()
        st.metric("Avg per Path", f"{avg_customers:.0f}")
    with col4:
        top_path = journey.loc[journey['customer_count'].idxmax()]
        st.metric("Most Common", f"{top_path['customer_count']:,} customers")
    
    st.markdown("---")
    
    # Sankey Diagram
    st.subheader("🌊 Customer Journey Flow (Sankey Diagram)")
    
    # Prepare data for Sankey
    def create_sankey_data(df):
        """Create nodes and links for Sankey diagram"""
        
        # Get all unique touchpoints
        all_touchpoints = set()
        for col in ['touchpoint_1', 'touchpoint_2', 'touchpoint_3', 'touchpoint_4']:
            all_touchpoints.update(df[col].dropna().unique())
        
        # Create node labels with stage prefix
        node_labels = []
        node_dict = {}
        
        for i, col in enumerate(['touchpoint_1', 'touchpoint_2', 'touchpoint_3', 'touchpoint_4'], 1):
            for tp in df[col].dropna().unique():
                label = f"{tp} (Stage {i})"
                if label not in node_labels:
                    node_labels.append(label)
                    node_dict[f"{col}_{tp}"] = len(node_labels) - 1
        
        # Create links
        sources = []
        targets = []
        values = []
        
        for _, row in df.iterrows():
            for i in range(3):
                source_col = f'touchpoint_{i+1}'
                target_col = f'touchpoint_{i+2}'
                
                if pd.notna(row[source_col]) and pd.notna(row[target_col]):
                    source_key = f"{source_col}_{row[source_col]}"
                    target_key = f"{target_col}_{row[target_col]}"
                    
                    if source_key in node_dict and target_key in node_dict:
                        sources.append(node_dict[source_key])
                        targets.append(node_dict[target_key])
                        values.append(row['customer_count'])
        
        return node_labels, sources, targets, values
    
    node_labels, sources, targets, values = create_sankey_data(journey)
    
    # Color mapping for touchpoints
    touchpoint_colors = {
        'Paid Search': COLORS['primary'],
        'Social Media': COLORS['secondary'],
        'Email': COLORS['success'],
        'Website': COLORS['info'],
        'Retargeting': COLORS['warning'],
        'Purchase': COLORS['purple'],
        'Organic Search': COLORS['teal'],
        'Display Ads': COLORS['pink']
    }
    
    node_colors = []
    for label in node_labels:
        base_touchpoint = label.split(' (Stage')[0]
        node_colors.append(touchpoint_colors.get(base_touchpoint, COLORS['gray']))
    
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="white", width=2),
            label=node_labels,
            color=node_colors
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color='rgba(100, 100, 100, 0.2)'
        )
    )])
    
    fig.update_layout(
        title="Customer Journey Flow - Touchpoint Progression",
        font=dict(size=10),
        height=700,
        template='plotly_white'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Touchpoint Analysis
    st.subheader("📊 Touchpoint Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### Touchpoint Frequency by Stage")
        
        # Count touchpoints at each stage
        touchpoint_counts = []
        for i, col in enumerate(['touchpoint_1', 'touchpoint_2', 'touchpoint_3', 'touchpoint_4'], 1):
            stage_counts = journey.groupby(col)['customer_count'].sum().reset_index()
            stage_counts['stage'] = f'Stage {i}'
            stage_counts.columns = ['touchpoint', 'count', 'stage']
            touchpoint_counts.append(stage_counts)
        
        touchpoint_df = pd.concat(touchpoint_counts, ignore_index=True)
        
        fig = px.bar(
            touchpoint_df,
            x='stage',
            y='count',
            color='touchpoint',
            title='Touchpoint Distribution Across Journey Stages',
            labels={'count': 'Customer Count', 'stage': 'Journey Stage', 'touchpoint': 'Touchpoint'},
            color_discrete_sequence=SEQUENTIAL_COLORS,
            barmode='stack'
        )
        
        fig.update_layout(
            height=450,
            template='plotly_white',
            legend=dict(orientation="v", yanchor="top", y=0.99, xanchor="left", x=1.02)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("##### Most Common Journey Paths")
        
        top_paths = journey.nlargest(10, 'customer_count').copy()
        top_paths['path'] = (
            top_paths['touchpoint_1'] + ' → ' +
            top_paths['touchpoint_2'] + ' → ' +
            top_paths['touchpoint_3'] + ' → ' +
            top_paths['touchpoint_4']
        )
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            y=top_paths['path'],
            x=top_paths['customer_count'],
            orientation='h',
            marker=dict(
                color=top_paths['customer_count'],
                colorscale='Blues',
                showscale=True,
                colorbar=dict(title="Customers")
            ),
            text=top_paths['customer_count'].apply(lambda x: f'{x:,}'),
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>Customers: %{x:,}<extra></extra>'
        ))
        
        fig.update_layout(
            title='Top 10 Customer Journey Paths',
            xaxis_title='Number of Customers',
            yaxis_title='',
            height=450,
            template='plotly_white',
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Journey Length Analysis
    st.subheader("📏 Journey Characteristics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### First Touchpoint Distribution")
        
        first_touch = journey.groupby('touchpoint_1')['customer_count'].sum().reset_index()
        first_touch = first_touch.sort_values('customer_count', ascending=False)
        
        fig = go.Figure(data=[go.Pie(
            labels=first_touch['touchpoint_1'],
            values=first_touch['customer_count'],
            hole=0.4,
            marker=dict(colors=SEQUENTIAL_COLORS),
            textinfo='label+percent',
            hovertemplate='<b>%{label}</b><br>Customers: %{value:,}<br>Percentage: %{percent}<extra></extra>'
        )])
        
        fig.update_layout(
            title='First Touchpoint Distribution',
            height=400,
            template='plotly_white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("##### Conversion Touchpoint (Last)")
        
        last_touch = journey.groupby('touchpoint_4')['customer_count'].sum().reset_index()
        last_touch = last_touch.sort_values('customer_count', ascending=True)
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            y=last_touch['touchpoint_4'],
            x=last_touch['customer_count'],
            orientation='h',
            marker_color=COLORS['success'],
            text=last_touch['customer_count'].apply(lambda x: f'{x:,}'),
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>Conversions: %{x:,}<extra></extra>'
        ))
        
        fig.update_layout(
            title='Final Touchpoint Before Conversion',
            xaxis_title='Number of Conversions',
            yaxis_title='',
            height=400,
            template='plotly_white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed Journey Table
    st.subheader("📋 Detailed Journey Paths")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        min_customers = st.slider(
            "Min Customers per Path",
            min_value=int(journey['customer_count'].min()),
            max_value=int(journey['customer_count'].max()),
            value=int(journey['customer_count'].quantile(0.5))
        )
    
    with col2:
        filtered_journey = journey[journey['customer_count'] >= min_customers].sort_values('customer_count', ascending=False)
        
        st.dataframe(
            filtered_journey.style.format({
                'customer_count': '{:,}'
            }).background_gradient(subset=['customer_count'], cmap='Blues'),
            use_container_width=True,
            height=400
        )
        
        st.caption(f"Showing {len(filtered_journey)} paths with ≥ {min_customers} customers")

# =============================================================================
# PAGE: ML MODEL EVALUATION
# =============================================================================
def page_ml_evaluation(data):
    """Machine Learning Model Evaluation"""
    st.title("🤖 ML Model Evaluation")
    st.markdown("### Lead Scoring Model Performance Analysis")
    
    leads = data['leads']
    feature_imp = data['feature_importance']
    learning = data['learning_curve']
    
    # Model Overview Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_leads = len(leads)
        st.metric("Total Leads", f"{total_leads:,}")
    with col2:
        actual_conversions = leads['actual_converted'].sum()
        st.metric("Actual Conversions", f"{actual_conversions:,}")
    with col3:
        conversion_rate = (actual_conversions / total_leads * 100)
        st.metric("Conversion Rate", f"{conversion_rate:.1f}%")
    with col4:
        avg_prob = leads['predicted_probability'].mean()
        st.metric("Avg Pred Probability", f"{avg_prob:.3f}")
    
    st.markdown("---")
    
    # Confusion Matrix & ROC Curve
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Confusion Matrix")
        
        threshold = st.slider(
            "Classification Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Adjust threshold to balance precision and recall"
        )
        
        y_true = leads['actual_converted']
        y_pred = (leads['predicted_probability'] >= threshold).astype(int)
        
        cm = confusion_matrix(y_true, y_pred)
        
        # Create confusion matrix with labels
        cm_labels = np.array([
            [f'TN<br>{cm[0,0]:,}', f'FP<br>{cm[0,1]:,}'],
            [f'FN<br>{cm[1,0]:,}', f'TP<br>{cm[1,1]:,}']
        ])
        
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=['Predicted: Not Converted', 'Predicted: Converted'],
            y=['Actual: Not Converted', 'Actual: Converted'],
            text=cm_labels,
            texttemplate='%{text}',
            textfont={"size": 16},
            colorscale='Blues',
            showscale=True,
            colorbar=dict(title="Count"),
            hovertemplate='%{y}<br>%{x}<br>Count: %{z:,}<extra></extra>'
        ))
        
        fig.update_layout(
            title=f'Confusion Matrix (Threshold: {threshold})',
            height=450,
            template='plotly_white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Calculate metrics
        tn, fp, fn, tp = cm.ravel()
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Accuracy", f"{accuracy:.3f}")
            st.metric("Precision", f"{precision:.3f}")
        with col_b:
            st.metric("Recall", f"{recall:.3f}")
            st.metric("F1-Score", f"{f1:.3f}")
        with col_c:
            st.metric("Specificity", f"{specificity:.3f}")
            st.metric("FPR", f"{1-specificity:.3f}")
    
    with col2:
        st.subheader("📈 ROC Curve")
        
        fpr, tpr, thresholds_roc = roc_curve(leads['actual_converted'], leads['predicted_probability'])
        roc_auc = auc(fpr, tpr)
        
        fig = go.Figure()
        
        # ROC Curve
        fig.add_trace(go.Scatter(
            x=fpr,
            y=tpr,
            mode='lines',
            name=f'ROC Curve (AUC = {roc_auc:.3f})',
            line=dict(color=COLORS['primary'], width=3),
            fill='tonexty',
            fillcolor='rgba(31, 119, 180, 0.2)',
            hovertemplate='FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>'
        ))
        
        # Random classifier line
        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            name='Random Classifier',
            line=dict(color='gray', width=2, dash='dash'),
            hovertemplate='Random<extra></extra>'
        ))
        
        # Current threshold point
        current_fpr = 1 - specificity
        current_tpr = recall
        
        fig.add_trace(go.Scatter(
            x=[current_fpr],
            y=[current_tpr],
            mode='markers',
            name=f'Current Threshold ({threshold})',
            marker=dict(size=15, color=COLORS['danger'], symbol='star'),
            hovertemplate=f'Threshold: {threshold}<br>FPR: {current_fpr:.3f}<br>TPR: {current_tpr:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=f'ROC Curve (AUC = {roc_auc:.3f})',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate (Recall)',
            height=450,
            template='plotly_white',
            legend=dict(x=0.6, y=0.1),
            xaxis=dict(range=[-0.05, 1.05]),
            yaxis=dict(range=[-0.05, 1.05])
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Additional ROC metrics
        st.info(f"""
        **ROC Analysis:**
        - AUC Score: {roc_auc:.3f}
        - Optimal Threshold: {thresholds_roc[np.argmax(tpr - fpr)]:.3f}
        - Current TPR: {current_tpr:.3f}
        - Current FPR: {current_fpr:.3f}
        """)
    
    # Precision-Recall Curve
    st.subheader("🎯 Precision-Recall Curve")
    
    precision_curve, recall_curve, thresholds_pr = precision_recall_curve(
        leads['actual_converted'],
        leads['predicted_probability']
    )
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=recall_curve,
        y=precision_curve,
        mode='lines',
        name='Precision-Recall Curve',
        line=dict(color=COLORS['success'], width=3),
        fill='tonexty',
        fillcolor='rgba(44, 160, 44, 0.2)',
        hovertemplate='Recall: %{x:.3f}<br>Precision: %{y:.3f}<extra></extra>'
    ))
    
    # Baseline
    baseline = actual_conversions / total_leads
    fig.add_hline(
        y=baseline,
        line_dash="dash",
        line_color="gray",
        annotation_text=f"Baseline: {baseline:.3f}",
        annotation_position="right"
    )
    
    fig.update_layout(
        title='Precision-Recall Curve',
        xaxis_title='Recall',
        yaxis_title='Precision',
        height=400,
        template='plotly_white',
        xaxis=dict(range=[-0.05, 1.05]),
        yaxis=dict(range=[-0.05, 1.05])
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Feature Importance
    st.subheader("🎯 Feature Importance Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        feature_imp_sorted = feature_imp.sort_values('importance', ascending=True)
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            y=feature_imp_sorted['feature'],
            x=feature_imp_sorted['importance'],
            orientation='h',
            error_x=dict(
                type='data',
                array=feature_imp_sorted['importance_std'],
                visible=True,
                color='rgba(0,0,0,0.3)'
            ),
            marker=dict(
                color=feature_imp_sorted['importance'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Importance")
            ),
            text=feature_imp_sorted['importance'].apply(lambda x: f'{x:.3f}'),
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>Importance: %{x:.3f} ± %{error_x.array:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title='Feature Importance with Standard Deviation',
            xaxis_title='Importance Score',
            yaxis_title='',
            height=600,
            template='plotly_white',
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("##### Top Features")
        
        top_features = feature_imp.nlargest(10, 'importance')
        
        st.dataframe(
            top_features.style.format({
                'importance': '{:.4f}',
                'importance_std': '{:.4f}'
            }).background_gradient(subset=['importance'], cmap='Greens'),
            use_container_width=True,
            height=600
        )
    
    # Learning Curve
    st.subheader("📚 Learning Curve Analysis")
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        show_std = st.checkbox("Show Standard Deviation", value=True)
    
    with col1:
        fig = go.Figure()
        
        # Training score
        fig.add_trace(go.Scatter(
            x=learning['training_size'],
            y=learning['train_score'],
            mode='lines+markers',
            name='Training Score',
            line=dict(color=COLORS['primary'], width=3),
            marker=dict(size=8),
            hovertemplate='Training Size: %{x:,}<br>Score: %{y:.3f}<extra></extra>'
        ))
        
        if show_std:
            fig.add_trace(go.Scatter(
                x=learning['training_size'].tolist() + learning['training_size'].tolist()[::-1],
                y=(learning['train_score'] + learning['train_score_std']).tolist() + 
                  (learning['train_score'] - learning['train_score_std']).tolist()[::-1],
                fill='toself',
                fillcolor='rgba(31, 119, 180, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                showlegend=False,
                hoverinfo='skip'
            ))
        
        # Validation score
        fig.add_trace(go.Scatter(
            x=learning['training_size'],
            y=learning['validation_score'],
            mode='lines+markers',
            name='Validation Score',
            line=dict(color=COLORS['success'], width=3),
            marker=dict(size=8),
            hovertemplate='Training Size: %{x:,}<br>Score: %{y:.3f}<extra></extra>'
        ))
        
        if show_std:
            fig.add_trace(go.Scatter(
                x=learning['training_size'].tolist() + learning['training_size'].tolist()[::-1],
                y=(learning['validation_score'] + learning['validation_score_std']).tolist() + 
                  (learning['validation_score'] - learning['validation_score_std']).tolist()[::-1],
                fill='toself',
                fillcolor='rgba(44, 160, 44, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                showlegend=False,
                hoverinfo='skip'
            ))
        
        fig.update_layout(
            title='Model Learning Curve',
            xaxis_title='Training Set Size',
            yaxis_title='Score',
            height=450,
            template='plotly_white',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            yaxis=dict(range=[0.4, 1.0])
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Prediction Distribution
    st.subheader("📊 Prediction Probability Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### Distribution by Actual Class")
        
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=leads[leads['actual_converted'] == 0]['predicted_probability'],
            name='Not Converted',
            marker_color=COLORS['danger'],
            opacity=0.7,
            nbinsx=50,
            hovertemplate='Probability: %{x:.3f}<br>Count: %{y}<extra></extra>'
        ))
        
        fig.add_trace(go.Histogram(
            x=leads[leads['actual_converted'] == 1]['predicted_probability'],
            name='Converted',
            marker_color=COLORS['success'],
            opacity=0.7,
            nbinsx=50,
            hovertemplate='Probability: %{x:.3f}<br>Count: %{y}<extra></extra>'
        ))
        
        fig.add_vline(
            x=threshold,
            line_dash="dash",
            line_color="black",
            annotation_text=f"Threshold: {threshold}",
            annotation_position="top"
        )
        
        fig.update_layout(
            title='Predicted Probability Distribution',
            xaxis_title='Predicted Probability',
            yaxis_title='Count',
            barmode='overlay',
            height=400,
            template='plotly_white',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("##### Calibration Analysis")
        
        # Create probability bins
        leads['prob_bin'] = pd.cut(leads['predicted_probability'], bins=10)
        calibration = leads.groupby('prob_bin').agg({
            'actual_converted': 'mean',
            'predicted_probability': 'mean',
            'lead_id': 'count'
        }).reset_index()
        calibration.columns = ['bin', 'actual_rate', 'predicted_rate', 'count']
        
        fig = go.Figure()
        
        # Perfect calibration line
        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            name='Perfect Calibration',
            line=dict(color='gray', width=2, dash='dash')
        ))
        
        # Actual calibration
        fig.add_trace(go.Scatter(
            x=calibration['predicted_rate'],
            y=calibration['actual_rate'],
            mode='markers+lines',
            name='Model Calibration',
            marker=dict(
                size=calibration['count']/10,
                color=COLORS['primary'],
                line=dict(width=1, color='white')
            ),
            line=dict(color=COLORS['primary'], width=2),
            hovertemplate='Predicted: %{x:.3f}<br>Actual: %{y:.3f}<br>Count: %{marker.size:.0f}<extra></extra>'
        ))
        
        fig.update_layout(
            title='Calibration Curve (Size: Sample Count)',
            xaxis_title='Mean Predicted Probability',
            yaxis_title='Actual Conversion Rate',
            height=400,
            template='plotly_white',
            xaxis=dict(range=[-0.05, 1.05]),
            yaxis=dict(range=[-0.05, 1.05])
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Lead Scoring Insights
    st.subheader("💡 Lead Scoring Insights")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        high_quality = len(leads[leads['predicted_probability'] >= 0.7])
        st.success(f"""
        **High Quality Leads**
        - Count: {high_quality:,}
        - Percentage: {high_quality/total_leads*100:.1f}%
        - Avg Conversion: {leads[leads['predicted_probability'] >= 0.7]['actual_converted'].mean()*100:.1f}%
        """)
    
    with col2:
        medium_quality = len(leads[(leads['predicted_probability'] >= 0.3) & (leads['predicted_probability'] < 0.7)])
        st.info(f"""
        **Medium Quality Leads**
        - Count: {medium_quality:,}
        - Percentage: {medium_quality/total_leads*100:.1f}%
        - Avg Conversion: {leads[(leads['predicted_probability'] >= 0.3) & (leads['predicted_probability'] < 0.7)]['actual_converted'].mean()*100:.1f}%
        """)
    
    with col3:
        low_quality = len(leads[leads['predicted_probability'] < 0.3])
        st.warning(f"""
        **Low Quality Leads**
        - Count: {low_quality:,}
        - Percentage: {low_quality/total_leads*100:.1f}%
        - Avg Conversion: {leads[leads['predicted_probability'] < 0.3]['actual_converted'].mean()*100:.1f}%
        """)


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
