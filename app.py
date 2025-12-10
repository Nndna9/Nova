
"""
app.py

Full Streamlit dashboard for NovaMart Marketing Analytics.
Requires: novamart.xlsx with sheets:
- campaign_performance
- customer_data
- product_sales
- lead_scoring_results
- feature_importance
- learning_curve
- geographic_data
- channel_attribution
- funnel_data
- customer_journey
- correlation_matrix

Run:
    pip install -r requirements.txt
    streamlit run app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import altair as alt
from sklearn.metrics import roc_curve, auc, confusion_matrix
import io
from typing import Dict

st.set_page_config(layout="wide", page_title="NovaMart Marketing Analytics", initial_sidebar_state="expanded")

# -----------------------
# Data Loading Utilities
# -----------------------
@st.cache_data
def load_sheets(path: str = "novamart.xlsx") -> Dict[str, pd.DataFrame]:
    try:
        sheets = pd.read_excel(path, sheet_name=None)
    except Exception as e:
        st.error(f"Could not load '{path}': {e}")
        return {}
    normalized = {}
    for name, df in sheets.items():
        key = name.strip().lower().replace(" ", "_")
        normalized[key] = df
    return normalized

def get(df_dict, name):
    return df_dict.get(name, None)

# Load data
DATA_PATH = "novamart.xlsx"
sheets = load_sheets(DATA_PATH)

campaign = get(sheets, "campaign_performance")
customer = get(sheets, "customer_data")
product_sales = get(sheets, "product_sales")
lead = get(sheets, "lead_scoring_results")
feature_importance = get(sheets, "feature_importance")
learning_curve = get(sheets, "learning_curve")
geo = get(sheets, "geographic_data")
attribution = get(sheets, "channel_attribution")
funnel = get(sheets, "funnel_data")
journey = get(sheets, "customer_journey")
corr = get(sheets, "correlation_matrix")

# Helper: safe message when sheet missing
def require(df, name):
    if df is None:
        st.warning(f"Sheet '{name}' not found in {DATA_PATH}. This page will show limited content.")
        return False
    return True

# Basic preprocessing helpers
def ensure_datetime(df, col_candidates):
    if df is None:
        return None
    for c in col_candidates:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c])
            return df, c
    # fallback: try first column
    try:
        df.iloc[:,0] = pd.to_datetime(df.iloc[:,0])
        return df, df.columns[0]
    except Exception:
        return df, None

# Sidebar navigation
st.sidebar.title("NovaMart Analytics")
page = st.sidebar.radio("Pages", [
    "Executive Overview",
    "Campaign Analytics",
    "Customer Insights",
    "Product Performance",
    "Geographic Analysis",
    "Attribution & Funnel",
    "ML Model Evaluation"
])

# -----------------------
# Executive Overview
# -----------------------
if page == "Executive Overview":
    st.title("Executive Overview — NovaMart")
    st.markdown("High-level KPIs and trends.")

    col1, col2, col3, col4 = st.columns(4)
    # KPIs
    if campaign is not None:
        revenue_col = next((c for c in campaign.columns if c.lower() in ["revenue","rev","total_revenue"]), None)
        conv_col = next((c for c in campaign.columns if c.lower() in ["conversions","conversion","convs"]), None)
        roas_col = next((c for c in campaign.columns if c.lower() == "roas"), None)
    else:
        revenue_col = conv_col = roas_col = None

    total_revenue = campaign[revenue_col].sum() if revenue_col and campaign is not None else np.nan
    total_conversions = campaign[conv_col].sum() if conv_col and campaign is not None else np.nan
    avg_roas = campaign[roas_col].mean() if roas_col and campaign is not None else np.nan
    cust_count = customer.shape[0] if customer is not None else np.nan

    col1.metric("Total Revenue", f"₹{total_revenue:,.0f}" if not np.isnan(total_revenue) else "N/A")
    col2.metric("Total Conversions", f"{int(total_conversions):,}" if not np.isnan(total_conversions) else "N/A")
    col3.metric("Avg ROAS", f"{avg_roas:.2f}" if not np.isnan(avg_roas) else "N/A")
    col4.metric("Customers", f"{int(cust_count):,}" if not np.isnan(cust_count) else "N/A")

    st.markdown("---")
    st.subheader("Revenue Trend")
    if not require(campaign, "campaign_performance"):
        st.stop()
    df = campaign.copy()
    df, date_col = ensure_datetime(df, ["date", "Date", "day"])
    if date_col is None:
        st.error("No valid date column found in campaign_performance.")
    else:
        agg = st.selectbox("Aggregate by", ["Daily", "Weekly", "Monthly"], index=2)
        if agg == "Daily":
            tmp = df.groupby(date_col, as_index=False).agg({revenue_col:'sum'})
            xcol = date_col
        elif agg == "Weekly":
            df['week'] = df[date_col].dt.to_period('W').apply(lambda r: r.start_time)
            tmp = df.groupby('week', as_index=False).agg({revenue_col:'sum'})
            xcol = 'week'
        else:
            df['month'] = df[date_col].dt.to_period('M').apply(lambda r: r.start_time)
            tmp = df.groupby('month', as_index=False).agg({revenue_col:'sum'})
            xcol = 'month'
        fig = px.line(tmp, x=xcol, y=revenue_col, title="Revenue Over Time")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Channel Performance")
    metric = st.selectbox("Metric", ["revenue","conversions","roas"], index=0)
    metric_col = next((c for c in df.columns if c.lower()==metric), None)
    if metric_col is None:
        st.info(f"Metric '{metric}' not found; trying fallback names.")
        metric_col = revenue_col if metric=="revenue" else conv_col if metric=="conversions" else roas_col
    if metric_col is None:
        st.error("Required metric column not found.")
    else:
        agg_ch = df.groupby('channel', as_index=False).agg({metric_col:'sum' if metric!='roas' else 'mean'}).sort_values(by=metric_col, ascending=True)
        fig = px.bar(agg_ch, x=metric_col, y='channel', orientation='h', title=f"Channel - {metric.capitalize()}")
        st.plotly_chart(fig, use_container_width=True)

# -----------------------
# Campaign Analytics
# -----------------------
if page == "Campaign Analytics":
    st.title("Campaign Analytics")
    if not require(campaign, "campaign_performance"):
        st.stop()
    df = campaign.copy()
    df, date_col = ensure_datetime(df, ["date","Date"])
    # 1.1 Bar Chart - Channel Performance Comparison
    st.header("Channel Performance Comparison")
    metric = st.selectbox("Choose metric for channel comparison", ["revenue","conversions","roas"], key="cmp_metric")
    metric_col = next((c for c in df.columns if c.lower()==metric), None)
    if metric_col is None:
        metric_col = revenue_col if metric=="revenue" else conv_col if metric=="conversions" else roas_col
    agg_channels = df.groupby('channel', as_index=False).agg({metric_col:'sum' if metric!='roas' else 'mean'}).sort_values(by=metric_col, ascending=True)
    fig = px.bar(agg_channels, x=metric_col, y='channel', orientation='h', title=f"Total {metric.capitalize()} by Channel")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    # 1.2 Grouped Bar Chart - Regional Performance by Quarter
    st.header("Regional Performance by Quarter")
    if 'region' not in df.columns:
        st.warning("No 'region' column. Skipping regional grouped bar chart.")
    else:
        df['quarter'] = df[date_col].dt.to_period('Q').astype(str)
        year_choice = st.selectbox("Year", sorted(df[date_col].dt.year.unique()), index=0)
        df_y = df[df[date_col].dt.year==int(year_choice)]
        grouped = df_y.groupby(['quarter','region'], as_index=False).agg({revenue_col:'sum'})
        fig = px.bar(grouped, x='quarter', y=revenue_col, color='region', barmode='group', title=f"Revenue by Region — {year_choice}")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    # 1.3 Stacked Bar Chart - Campaign Type Contribution
    st.header("Monthly Spend by Campaign Type (Stacked)")
    if 'campaign_type' in df.columns and 'spend' in df.columns:
        toggle = st.radio("Stack mode", ["Absolute","100%"], index=0)
        monthly = df.copy()
        monthly['month'] = monthly[date_col].dt.to_period('M').apply(lambda r: r.start_time)
        pivot = monthly.groupby(['month','campaign_type'], as_index=False).agg({'spend':'sum'})
        if toggle == "100%":
            pivot_total = pivot.groupby('month', as_index=False).agg({'spend':'sum'}).rename(columns={'spend':'total'})
            merged = pivot.merge(pivot_total, on='month')
            merged['pct'] = merged['spend']/merged['total']
            fig = px.bar(merged, x='month', y='pct', color='campaign_type', title="Monthly Spend (100% stacked)")
        else:
            fig = px.bar(pivot, x='month', y='spend', color='campaign_type', title="Monthly Spend by Campaign Type")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Columns 'campaign_type' or 'spend' not found for stacked chart.")

    st.markdown("---")
    # 2.1 Line Chart - Revenue Trend Over Time with channel breakdown
    st.header("Revenue Trend (Channel breakdown)")
    agg_level = st.selectbox("Aggregation", ["Daily","Weekly","Monthly"], key="agg_level_trend")
    channels = df['channel'].unique().tolist() if 'channel' in df.columns else []
    sel_channels = st.multiselect("Channels", options=channels, default=channels[:3])
    if agg_level=="Daily":
        tmp = df.groupby([date_col,'channel'], as_index=False).agg({revenue_col:'sum'})
        tmp = tmp[tmp['channel'].isin(sel_channels)]
        fig = px.line(tmp, x=date_col, y=revenue_col, color='channel', title="Daily Revenue by Channel")
    elif agg_level=="Weekly":
        df['week'] = df[date_col].dt.to_period('W').apply(lambda r: r.start_time)
        tmp = df.groupby(['week','channel'], as_index=False).agg({revenue_col:'sum'})
        tmp = tmp[tmp['channel'].isin(sel_channels)]
        fig = px.line(tmp, x='week', y=revenue_col, color='channel', title="Weekly Revenue by Channel")
    else:
        df['month'] = df[date_col].dt.to_period('M').apply(lambda r: r.start_time)
        tmp = df.groupby(['month','channel'], as_index=False).agg({revenue_col:'sum'})
        tmp = tmp[tmp['channel'].isin(sel_channels)]
        fig = px.line(tmp, x='month', y=revenue_col, color='channel', title="Monthly Revenue by Channel")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    # 2.2 Area Chart - Cumulative Conversions by channel
    st.header("Cumulative Conversions by Channel")
    if conv_col is not None:
        region_filter = st.selectbox("Region filter", options=['All'] + df['region'].dropna().unique().tolist())
        dfc = df.copy()
        if region_filter != 'All' and 'region' in dfc.columns:
            dfc = dfc[dfc['region']==region_filter]
        dfc = dfc.sort_values(by=date_col)
        dfc['cum_conv'] = dfc.groupby('channel')[conv_col].cumsum()
        tmp = dfc.groupby([date_col,'channel'], as_index=False).agg({conv_col:'sum'})
        tmp = tmp.sort_values(by=date_col)
        fig = px.area(dfc, x=date_col, y='cum_conv', color='channel', title="Cumulative Conversions")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No conversions column found for cumulative conversions.")

    st.markdown("---")
    # Calendar heatmap (simple representation)
    st.header("Calendar Heatmap — Daily Revenue Intensity")
    years = sorted(df[date_col].dt.year.unique().tolist())
    year_choice = st.selectbox("Year", years, index=len(years)-1)
    dfy = df[df[date_col].dt.year==int(year_choice)]
    daily = dfy.groupby(dfy[date_col].dt.date, as_index=False).agg({revenue_col:'sum'}).rename(columns={date_col:'date', revenue_col:'revenue'})
    # Use heatmap by day of year (simple)
    daily['dayofyear'] = pd.to_datetime(daily['date']).dt.dayofyear
    daily['weekday'] = pd.to_datetime(daily['date']).dt.weekday
    pivot = daily.pivot_table(index='weekday', columns='dayofyear', values='revenue', fill_value=0)
    fig = go.Figure(data=go.Heatmap(z=pivot.values, x=pivot.columns, y=['Mon','Tue','Wed','Thu','Fri','Sat','Sun']))
    fig.update_layout(title=f"Daily Revenue Heatmap — {year_choice}", xaxis_title='Day of Year')
    st.plotly_chart(fig, use_container_width=True)

# -----------------------
# Customer Insights
# -----------------------
if page == "Customer Insights":
    st.title("Customer Insights")
    if not require(customer, "customer_data"):
        st.stop()
    df = customer.copy()
    # Histogram - Age distribution
    st.header("Customer Age Distribution")
    age_col = next((c for c in df.columns if 'age' in c.lower()), None)
    if age_col:
        bins = st.slider("Bins", 5, 50, 15)
        seg = st.selectbox("Segment filter (optional)", options=['All'] + df['segment'].dropna().unique().tolist())
        dff = df if seg=='All' else df[df['segment']==seg]
        fig = px.histogram(dff, x=age_col, nbins=bins, title="Age Distribution")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No age column found.")

    st.markdown("---")
    # Box plot - LTV by segment
    st.header("Lifetime Value by Customer Segment")
    ltv_col = next((c for c in df.columns if 'ltv' in c.lower()), None)
    seg_col = next((c for c in df.columns if 'segment' in c.lower()), None)
    if ltv_col and seg_col:
        show_points = st.checkbox("Show points", value=False)
        fig = px.box(df, x=seg_col, y=ltv_col, points='all' if show_points else 'outliers', title="LTV by Segment")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("LTV or Segment column missing.")

    st.markdown("---")
    # Violin - Satisfaction by NPS
    st.header("Satisfaction Score Distribution by NPS")
    sat_col = next((c for c in df.columns if 'satisf' in c.lower()), None)
    nps_col = next((c for c in df.columns if 'nps' in c.lower() or 'promoter' in c.lower()), None)
    if sat_col and nps_col:
        split_acq = st.checkbox("Split by acquisition channel", value=False)
        if split_acq and 'acquisition_channel' in df.columns:
            fig = px.violin(df, x=nps_col, y=sat_col, color='acquisition_channel', box=True, points='all')
        else:
            fig = px.violin(df, x=nps_col, y=sat_col, box=True, points='all')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Satisfaction or NPS column not found.")

    st.markdown("---")
    # Scatter Income vs LTV
    st.header("Income vs Lifetime Value")
    income_col = next((c for c in df.columns if 'income' in c.lower()), None)
    if income_col and ltv_col:
        fig = px.scatter(df, x=income_col, y=ltv_col, color=seg_col if seg_col else None, hover_data=df.columns, trendline="ols")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Income or LTV column missing for scatter plot.")

# -----------------------
# Product Performance
# -----------------------
if page == "Product Performance":
    st.title("Product Performance")
    if not require(product_sales, "product_sales"):
        st.stop()
    df = product_sales.copy()
    # Treemap
    st.header("Product Hierarchy Treemap")
    # expect columns: Category, Subcategory, Product, Sales, Profit_Margin
    cat = next((c for c in df.columns if 'category' in c.lower()), None)
    sub = next((c for c in df.columns if 'sub' in c.lower()), None)
    prod = next((c for c in df.columns if 'product' in c.lower()), None)
    sales = next((c for c in df.columns if 'sale' in c.lower()), None)
    profit = next((c for c in df.columns if 'profit' in c.lower()), None)
    if cat and sub and prod and sales:
        fig = px.treemap(df, path=[cat, sub, prod], values=sales, color=profit if profit else sales, title="Sales Treemap (size=Sales, color=Profit)")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Expected product_sales columns missing (Category/Subcategory/Product/Sales).")

    st.markdown("---")
    st.header("Category Comparison")
    if sales:
        agg = df.groupby(cat if cat else df.columns[0], as_index=False).agg({sales:'sum'}).sort_values(by=sales, ascending=False)
        fig = px.bar(agg, x=cat if cat else agg.columns[0], y=sales, title="Sales by Category")
        st.plotly_chart(fig, use_container_width=True)

# -----------------------
# Geographic Analysis
# -----------------------
if page == "Geographic Analysis":
    st.title("Geographic Analysis")
    if not require(geo, "geographic_data"):
        st.stop()
    df = geo.copy()
    # Choropleth - requires state and revenue
    state_col = next((c for c in df.columns if 'state' in c.lower()), None)
    lat_col = next((c for c in df.columns if 'lat' in c.lower()), None)
    lon_col = next((c for c in df.columns if 'lon' in c.lower() or 'lng' in c.lower()), None)
    rev_col = next((c for c in df.columns if 'revenue' in c.lower()), None)

    st.header("State-wise Revenue (Bubble Map)")
    metric = st.selectbox("Metric", options=[rev_col, 'customers','market_penetration','yoY_growth'] if rev_col else df.columns.tolist(), format_func=lambda x: x)
    if lat_col and lon_col:
        fig = px.scatter_mapbox(df, lat=lat_col, lon=lon_col, size=rev_col if rev_col else None,
                                hover_name=state_col, hover_data=df.columns,
                                zoom=3, height=600)
        fig.update_layout(mapbox_style="open-street-map")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No latitude/longitude columns found; showing choropleth substitute table.")
        st.dataframe(df.sort_values(by=rev_col, ascending=False).head(20))

# -----------------------
# Attribution & Funnel
# -----------------------
if page == "Attribution & Funnel":
    st.title("Attribution & Funnel")
    if require(attribution, "channel_attribution"):
        df = attribution.copy()
        # Donut chart for attribution models
        st.header("Attribution Model Comparison")
        models = [c for c in df.columns if c.lower() in ['first_touch','last_touch','linear','time_decay','position_based'] or 'first' in c.lower()]
        if len(models)==0:
            models = df.columns[1:].tolist() if df.shape[1]>1 else df.columns.tolist()
        model_choice = st.selectbox("Model", options=models)
        # assume channels in first column
        channel_col = df.columns[0]
        fig = px.pie(df, names=channel_col, values=model_choice, title=f"Attribution - {model_choice}", hole=0.45)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    if require(funnel, "funnel_data"):
        st.header("Funnel Chart")
        ff = funnel.copy()
        # expect columns Stage and Count
        stage_col = next((c for c in ff.columns if 'stage' in c.lower()), ff.columns[0])
        count_col = next((c for c in ff.columns if 'count' in c.lower() or 'visitors' in c.lower()), ff.columns[1] if ff.shape[1]>1 else ff.columns[0])
        funnel_order = ff[stage_col].tolist()
        funnel_counts = ff[count_col].tolist()
        fig = go.Figure(go.Funnel(y=funnel_order, x=funnel_counts))
        st.plotly_chart(fig, use_container_width=True)

    if require(corr, "correlation_matrix"):
        st.header("Correlation Heatmap")
        cm = corr.copy()
        # If correlation matrix is provided as table with index
        try:
            fig = px.imshow(cm.values, x=cm.columns, y=cm.columns, color_continuous_scale='RdBu', zmin=-1, zmax=1)
            fig.update_layout(title="Correlation Matrix")
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            st.info("Correlation matrix could not be rendered as heatmap.")

# -----------------------
# ML Model Evaluation
# -----------------------
if page == "ML Model Evaluation":
    st.title("ML Model Evaluation — Lead Scoring")
    if not require(lead, "lead_scoring_results"):
        st.stop()
    df = lead.copy()
    # Ensure columns for actual and predicted prob
    actual_col = next((c for c in df.columns if c.lower() in ['actual_converted','actual','converted','is_converted']), None)
    prob_col = next((c for c in df.columns if 'prob' in c.lower() or 'predicted_probability' in c.lower()), None)
    pred_col = next((c for c in df.columns if c.lower() in ['predicted_class','pred_class','predicted']), None)

    if prob_col and actual_col:
        st.header("ROC Curve")
        y_true = df[actual_col].astype(int)
        y_score = df[prob_col].astype(float)
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC curve (AUC = {roc_auc:.3f})'))
        fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', name='Random', line=dict(dash='dash')))
        st.plotly_chart(fig, use_container_width=True)
        # Threshold slider and confusion matrix
        thr = st.slider("Threshold", 0.0, 1.0, 0.5)
        preds = (y_score >= thr).astype(int)
        cm = confusion_matrix(y_true, preds)
        st.subheader("Confusion Matrix (counts)")
        st.write(cm)
        st.subheader("Confusion Matrix (percentages)")
        cm_pct = cm / cm.sum()
        st.write(np.round(cm_pct,3))
    else:
        st.info("Required columns for ROC not found (actual/probability).")

    if feature_importance is not None:
        st.header("Feature Importance")
        fi = feature_importance.copy()
        # expect columns 'feature' and 'importance'
        feat_col = fi.columns[0]
        imp_col = fi.columns[1] if fi.shape[1]>1 else fi.columns[0]
        fi_sorted = fi.sort_values(by=imp_col, ascending=True)
        fig = px.bar(fi_sorted, x=imp_col, y=feat_col, orientation='h', title="Feature Importance (with error bars if present)")
        st.plotly_chart(fig, use_container_width=True)

    if learning_curve is not None:
        st.header("Learning Curve")
        lc = learning_curve.copy()
        # expect columns train_size, train_score, val_score, and stds optional
        if 'train_size' in lc.columns:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=lc['train_size'], y=lc['train_score'], mode='lines+markers', name='Train'))
            fig.add_trace(go.Scatter(x=lc['train_size'], y=lc['val_score'], mode='lines+markers', name='Validation'))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("learning_curve sheet lacks expected columns (train_size).")

# End of app
