# NovaMart Streamlit Dashboard

This repository contains a Streamlit dashboard for NovaMart marketing analytics.  
Place `novamart.xlsx` (with the 11 required sheets) in the same directory as `app.py`.

## Sheets expected in `novamart.xlsx` (sheet names should be similar):
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

## Quickstart

1. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   venv\Scripts\activate   # Windows
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the app:
   ```bash
   streamlit run app.py
   ```

## Notes
- The app attempts to detect common column names. If a chart does not render, check sheet column headers for expected names (e.g., 'date', 'revenue', 'channel', 'spend', 'conversions', 'region').
- Maps use latitude/longitude columns; if missing, a fallback table is shown.
- The app focuses on interactivity and modular structure; you can extend each section by editing functions in `app.py`.
