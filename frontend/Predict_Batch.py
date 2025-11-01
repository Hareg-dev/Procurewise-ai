"""
Batch prediction page for CSV file upload and bulk risk assessment.
"""
import streamlit as st
import pandas as pd
import io
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from components.layout import setup_page_config, render_header, render_sidebar, render_footer, show_loading, show_success, show_error, show_info
from components.charts import plot_batch_results_summary
from utils.api_client import APIClient

def create_sample_csv():
    """Create sample CSV data for download."""
    sample_data = {
        'supplier_id': ['S_001', 'S_002', 'S_003'],
        'annual_revenue': [1000000, 2500000, 750000],
        'annual_spend': [500000, 1200000, 400000],
        'avg_payment_delay_days': [15, 25, 8],
        'contract_value': [250000, 600000, 180000],
        'contract_duration_months': [12, 24, 18],
        'past_disputes': [2, 5, 1],
        'delivery_score': [85, 70, 92],
        'financial_stability_index': [0.75, 0.60, 0.85],
        'relationship_years': [5, 8, 3],
        'credit_score': [720, 650, 780],
        'txn_count': [10, 25, 8],
        'avg_txn_amount': [25000, 48000, 22500],
        'avg_delay': [12, 20, 6],
        'late_ratio': [0.2, 0.4, 0.1],
        'dispute_rate': [0.1, 0.2, 0.05],
        'avg_delivery_quality': [88, 75, 95],
        'clause_risk_score': [25, 45, 15],
        'region': ['North America', 'Europe', 'Asia-Pacific'],
        'industry': ['IT Services', 'Manufacturing', 'Biotech'],
        'contract_criticality': ['Medium', 'High', 'Low']
    }
    return pd.DataFrame(sample_data)

def display_csv_requirements():
    """Display CSV format requirements."""
    with st.expander("📋 CSV Format Requirements", expanded=False):
        st.markdown("""
        **Required Columns:**
        
        **Financial Data:**
        - `annual_revenue`: Annual revenue in USD
        - `annual_spend`: Annual spend in USD  
        - `contract_value`: Contract value in USD
        - `credit_score`: Credit score (300-850)
        - `financial_stability_index`: Financial stability (0.0-1.0)
        
        **Performance Data:**
        - `delivery_score`: Delivery performance (0-100)
        - `avg_payment_delay_days`: Average payment delay in days
        - `avg_delivery_quality`: Average delivery quality (0-100)
        - `past_disputes`: Number of past disputes
        
        **Relationship Data:**
        - `relationship_years`: Years of relationship
        - `contract_duration_months`: Contract duration in months
        - `clause_risk_score`: Contract clause risk score (0-100)
        
        **Transaction Data:**
        - `txn_count`: Transaction count
        - `avg_txn_amount`: Average transaction amount
        - `avg_delay`: Average delay in days
        - `late_ratio`: Late payment ratio (0.0-1.0)
        - `dispute_rate`: Dispute rate (0.0-1.0)
        
        **Categorical Data:**
        - `region`: Geographic region
        - `industry`: Industry sector
        - `contract_criticality`: Contract criticality level
        
        **Optional:**
        - `supplier_id`: Unique supplier identifier
        """)

def display_batch_results(result_data):
    """Display batch prediction results with visualizations."""
    st.markdown("## 📊 Batch Prediction Results")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", result_data["total_records"])
    
    with col2:
        st.metric("Successful Predictions", result_data["successful_predictions"])
    
    with col3:
        st.metric("Failed Predictions", result_data["failed_predictions"])
    
    with col4:
        st.metric("Avg Confidence", f"{result_data['avg_confidence']:.1%}")
    
    # Risk distribution
    st.markdown("### 🎯 Risk Distribution")
    risk_dist = result_data["risk_distribution"]
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🔴 High Risk", risk_dist["high"])
    with col2:
        st.metric("🟡 Medium Risk", risk_dist["medium"])
    with col3:
        st.metric("🟢 Low Risk", risk_dist["low"])
    
    # Detailed results table
    st.markdown("### 📋 Detailed Results")
    
    results_df = pd.DataFrame(result_data["results"])
    
    if not results_df.empty:
        # Filter and display options
        col1, col2 = st.columns(2)
        
        with col1:
            risk_filter = st.selectbox(
                "Filter by Risk Level",
                ["All", "high", "medium", "low"],
                key="risk_filter"
            )
        
        with col2:
            show_errors = st.checkbox("Show Failed Predictions", value=False)
        
        # Apply filters
        filtered_df = results_df.copy()
        
        if risk_filter != "All":
            filtered_df = filtered_df[filtered_df['prediction'] == risk_filter]
        
        if not show_errors:
            filtered_df = filtered_df[filtered_df['prediction'] != 'Error']
        
        # Display results
        st.dataframe(
            filtered_df,
            use_container_width=True,
            hide_index=True
        )
        
        # Visualizations
        if len(filtered_df) > 0 and 'prediction' in filtered_df.columns:
            plot_batch_results_summary(filtered_df)
        
        # Download results
        st.markdown("### 💾 Download Results")
        
        csv_buffer = io.StringIO()
        results_df.to_csv(csv_buffer, index=False)
        csv_data = csv_buffer.getvalue()
        
        st.download_button(
            label="📥 Download Results as CSV",
            data=csv_data,
            file_name=f"batch_predictions_{result_data['batch_id'][:8]}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    else:
        st.error("No results to display")

def main():
    setup_page_config()
    render_header()
    render_sidebar()
    
    st.markdown("# 📊 Batch Risk Prediction")
    st.markdown("Upload a CSV file to analyze multiple suppliers at once and get comprehensive risk insights.")
    
    # CSV requirements
    display_csv_requirements()
    
    # Sample CSV download
    st.markdown("### 📥 Sample CSV Template")
    sample_df = create_sample_csv()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("Download a sample CSV file to see the required format:")
        csv_sample = sample_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Sample CSV",
            data=csv_sample,
            file_name="sample_suppliers.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        st.markdown("Preview of sample data:")
        st.dataframe(sample_df.head(3), use_container_width=True)
    
    st.markdown("---")
    
    # File upload
    st.markdown("### 📤 Upload Your CSV File")
    
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="Upload a CSV file with supplier data for batch prediction"
    )
    
    if uploaded_file is not None:
        try:
            # Preview uploaded data
            df = pd.read_csv(uploaded_file)
            
            st.markdown("### 👀 Data Preview")
            st.markdown(f"**File:** {uploaded_file.name}")
            st.markdown(f"**Records:** {len(df)}")
            st.markdown(f"**Columns:** {len(df.columns)}")
            
            # Show preview
            st.dataframe(df.head(10), use_container_width=True)
            
            # Validation
            required_columns = [
                'annual_revenue', 'annual_spend', 'avg_payment_delay_days',
                'contract_value', 'contract_duration_months', 'past_disputes',
                'delivery_score', 'financial_stability_index', 'relationship_years',
                'credit_score', 'region', 'industry', 'contract_criticality'
            ]
            
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                show_error(f"Missing required columns: {', '.join(missing_columns)}")
                st.markdown("Please ensure your CSV contains all required columns as shown in the requirements above.")
            else:
                show_success("CSV format validation passed!")
                
                # Prediction button
                if st.button("🚀 Start Batch Prediction", use_container_width=True):
                    api_client = APIClient()
                    
                    # Reset file pointer
                    uploaded_file.seek(0)
                    
                    with show_loading(f"Processing {len(df)} suppliers..."):
                        result = api_client.predict_batch(uploaded_file)
                    
                    if result["success"]:
                        show_success("Batch prediction completed successfully!")
                        display_batch_results(result["data"])
                        
                        # Store in session state
                        if 'batch_results' not in st.session_state:
                            st.session_state.batch_results = []
                        st.session_state.batch_results.append(result["data"])
                        
                    else:
                        show_error(f"Batch prediction failed: {result['error']}")
                        st.markdown("**Troubleshooting:**")
                        st.markdown("- Check if the API server is running")
                        st.markdown("- Verify CSV format matches requirements")
                        st.markdown("- Ensure file size is reasonable (<10MB)")
        
        except Exception as e:
            show_error(f"Error reading CSV file: {str(e)}")
            st.markdown("Please check that your file is a valid CSV format.")
    
    else:
        show_info("Please upload a CSV file to begin batch prediction.")
    
    render_footer()

if __name__ == "__main__":
    main()