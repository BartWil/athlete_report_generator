import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import zscore
import matplotlib.pyplot as plt
import io
import base64
from jinja2 import Template
import os
from datetime import datetime

st.set_page_config(page_title="Athlete Functional Report Generator", layout="wide")

# Set title and description
st.title("Athlete Functional Report Generator")
st.markdown("Upload your Excel data file to generate personalized athlete functional reports.")

# File uploader
uploaded_file = st.file_uploader("Choose an Excel file", type=['xlsx', 'xls'])

# Sidebar
st.sidebar.header("About")
st.sidebar.info(
    "This tool analyzes athlete functional assessment data and generates personalized reports "
    "with functional profile classification, Z-scores with radar charts, "
    "domain flags, and specific recommendations."
)

# Add export options in sidebar
st.sidebar.header("Quick Links")
if st.session_state.processed and st.session_state.df is not None:
    # Export data
    st.sidebar.download_button(
        label="📊 Export Processed Data (CSV)",
        data=st.session_state.df.to_csv(index=False).encode('utf-8'),
        file_name="processed_athlete_data.csv",
        mime="text/csv"
    )
    
    # Summary statistics
    st.sidebar.header("Summary Statistics")
    profile_counts = st.session_state.df['Deficit_Profile'].value_counts()
    flag_counts = st.session_state.df['Flag_Category'].value_counts()
    
    # Display pie chart of profiles
    profile_data = pd.DataFrame({
        'Profile': profile_counts.index,
        'Count': profile_counts.values
    })
    
    st.sidebar.markdown("### Functional Profiles")
    st.sidebar.dataframe(profile_data)
    
    # Display flag distribution
    st.sidebar.markdown("### Risk Levels")
    st.sidebar.dataframe(flag_counts)

# Initialize session state variables if they don't exist
if 'df' not in st.session_state:
    st.session_state.df = None
if 'processed' not in st.session_state:
    st.session_state.processed = False
if 'column_mapping' not in st.session_state:
    st.session_state.column_mapping = {
        'HAbd_L PEAK FORCE (KG) Normalized to body weight Raw': 'HAbd_L',
        'HAbd_P PEAK FORCE (KG) Normalized to body weight Raw': 'HAbd_P',
        'KE_L PEAK FORCE (KG) Normalized to body weight Raw': 'KE_L',
        'KE_P PEAK FORCE (KG) Normalized to body weight Raw': 'KE_P',
        'KF_L PEAK FORCE(KG) Normalized to body weight Raw': 'KF_L',
        'KF_P PEAK FORCE (KG) Normalized to body weight Raw': 'KF_P',
        'AP_L PEAK FORCE (KG) Normalized to body weight Raw': 'AP_L',
        'AP_P_PEAK_FORCE_(KG)_Normalized_to_body_weight Raw': 'AP_P',
        'YBT_ANT_L_Normalized': 'YBT_ANT_L',
        'YBT_ANT_R_Normalized': 'YBT_ANT_P',
        'YBT_PM_L_Normalized': 'YBT_PM_L',
        'YBT_PM_R_Normalized': 'YBT_PM_P',
        'YBT_PL_L_Normalized': 'YBT_PL_L',
        'YBT_PL_R_Normalized': 'YBT_PL_P'
    }
if 'limb_columns' not in st.session_state:
    st.session_state.limb_columns = {
        'HAbd': ('HAbd_L', 'HAbd_P'),
        'KE': ('KE_L', 'KE_P'),
        'KF': ('KF_L', 'KF_P'),
        'AP': ('AP_L', 'AP_P'),
        'YBT ANT': ('YBT_ANT_L', 'YBT_ANT_P'),
        'YBT PM': ('YBT_PM_L', 'YBT_PM_P'),
        'YBT PL': ('YBT_PL_L', 'YBT_PL_P'),
    }
if 'profile_order' not in st.session_state:
    st.session_state.profile_order = [
        'Functionally weak',
        'Strength-deficient',
        'Stability-deficient',
        'No clear dysfunction'
    ]

# Helper functions
def fms_category(percentile):
    """Categorize FMS percentile"""
    if percentile < 25:
        return 'Low'
    elif percentile < 75:
        return 'Medium'
    else:
        return 'High'

def classify_deficit_profile(row):
    """Classify athlete's deficit profile based on FMS, strength, and stability"""
    fms_cat = row['FMS_category']
    str_z = row['Mean_Strength_Z']
    ybt_z = row['Mean_YBT_Z']
    
    # 1. Functionally weak
    if fms_cat == 'Low':
        return 'Functionally weak'
    # 2. Strength-deficient – only if FMS is not Low
    if str_z <= -0.5 and ybt_z > -0.5:
        return 'Strength-deficient'
    # 3. Stability-deficient – only if FMS and strength are normal
    if ybt_z <= -0.5 and str_z > -0.5:
        return 'Stability-deficient'
    # 4. Everyone else
    return 'No clear dysfunction'

def interpret_flags(score):
    """Interpret flag scores as Red, Yellow, or Green"""
    if score == 3:
        return "Red flag"
    elif score == 2:
        return "Yellow flag"
    else:
        return "Green flag"

def process_data(df):
    """Process and prepare data for analysis"""
    # Standardize column names
    available_cols = set(df.columns)
    for old_col, new_col in st.session_state.column_mapping.items():
        # Exact match
        if old_col in available_cols:
            df = df.rename(columns={old_col: new_col})
        else:
            # Try case-insensitive partial matching
            for col in available_cols:
                # Check if the important parts of the column name match
                if old_col.split('(')[0].strip().lower() in col.lower():
                    df = df.rename(columns={col: new_col})
                    break
    
    # Required columns
    required_strength_cols = ['HAbd_L', 'HAbd_P', 'KE_L', 'KE_P', 'KF_L', 'KF_P', 'AP_L', 'AP_P']
    required_ybt_cols = ['YBT_ANT_L', 'YBT_ANT_P', 'YBT_PM_L', 'YBT_PM_P', 'YBT_PL_L', 'YBT_PL_P']
    required_other_cols = ['BMI', 'Chronologic_Age', 'FMS_TOTAL']
    
    # Check required columns
    all_required = required_strength_cols + required_ybt_cols + required_other_cols
    missing = [col for col in all_required if col not in df.columns]
    
    if missing:
        st.error(f"Missing required columns: {', '.join(missing)}")
        return None
    
    # Calculate aggregates
    df['Mean_Strength'] = df[required_strength_cols].mean(axis=1)
    df['Mean_YBT'] = df[required_ybt_cols].mean(axis=1)
    
    # Z-score normalization
    df['Mean_Strength_Z'] = zscore(df['Mean_Strength'])
    df['Mean_YBT_Z'] = zscore(df['Mean_YBT'])
    df['BMI_Z'] = zscore(df['BMI'])
    df['Chronologic_Age_Z'] = zscore(df['Chronologic_Age'])
    
    # FMS percentiles
    df['FMS_percentile'] = df['FMS_TOTAL'].rank(pct=True) * 100
    df['FMS_category'] = df['FMS_percentile'].apply(fms_category)
    
    # Expert classification
    df['Deficit_Profile'] = df.apply(classify_deficit_profile, axis=1)
    
    # Flag calculation
    tercile_fms = df['FMS_TOTAL'].quantile(1/3)
    tercile_strength = df['Mean_Strength'].quantile(1/3)
    tercile_ybt = df['Mean_YBT'].quantile(1/3)
    
    df['flag_fms'] = (df['FMS_TOTAL'] <= tercile_fms).astype(int)
    df['flag_strength'] = (df['Mean_Strength'] <= tercile_strength).astype(int)
    df['flag_ybt'] = (df['Mean_YBT'] <= tercile_ybt).astype(int)
    
    df['Total_Flag_Score'] = df['flag_fms'] + df['flag_strength'] + df['flag_ybt']
    df['Flag_Category'] = df['Total_Flag_Score'].apply(interpret_flags)
    
    return df

def get_athlete_names(df):
    """Return a list of athlete names/identifiers"""
    if df is None:
        return []
    
    # Try to find name columns
    name_cols = [col for col in df.columns if 'name' in col.lower()]
    if name_cols:
        names = df[name_cols[0]].tolist()
    else:
        # If no name column, create generic identifiers
        names = [f"Athlete #{i}" for i in range(len(df))]
    
    return names

def generate_radar_chart(df, athlete_index):
    """Generate radar chart for an athlete"""
    if df is None or athlete_index >= len(df):
        return None
    
    athlete = df.iloc[athlete_index]
    
    # Prepare Z-score normalized data for left and right limbs
    left_z, right_z, labels = [], [], []
    for label, (left_col, right_col) in st.session_state.limb_columns.items():
        # Check if columns exist
        if left_col in df.columns and right_col in df.columns:
            left_z.append(zscore(df[left_col])[athlete_index])
            right_z.append(zscore(df[right_col])[athlete_index])
            labels.append(label)
    
    # If no valid data, return None
    if not left_z:
        return None
    
    # Close the radar chart loop
    left_z += left_z[:1]
    right_z += right_z[:1]
    labels += labels[:1]
    angles = np.linspace(0, 2 * np.pi, len(labels)-1, endpoint=False).tolist()
    angles += angles[:1]
    
    # Plot radar chart with background zones
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    
    # Background – risk zones
    for r, color in [(-3, '#ffe5e5'), (-1, '#fff2cc'), (0, '#e6ffe6')]:
        ax.fill_between(angles, r, r + 1.5, color=color, alpha=0.3)
    
    # Left limb – dark blue
    ax.plot(angles, left_z, label='Left', color='#1f77b4')
    ax.fill(angles, left_z, alpha=0.2, color='#1f77b4')
    
    # Right limb – purple
    ax.plot(angles, right_z, label='Right', color='#9467bd')
    ax.fill(angles, right_z, alpha=0.2, color='#9467bd')
    
    # Labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels[:-1])
    
    athlete_names = get_athlete_names(df)
    athlete_name = athlete_names[athlete_index] if athlete_index < len(athlete_names) else f"Athlete #{athlete_index}"
    ax.set_title(f"Z-score Functional Profile: Left vs Right Limb\n{athlete_name}", fontsize=13)
    ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
    plt.tight_layout()
    
    return fig

def generate_recommendation(df, athlete_index):
    """Generate personalized recommendation based on athlete's profile"""
    if df is None or athlete_index >= len(df):
        return ""
    
    athlete = df.iloc[athlete_index]
    profile = athlete['Deficit_Profile']
    flags = athlete['Flag_Category']
    
    # Base recommendations on profile and flags
    recommendation = ""
    
    # Profile-based recommendations
    if profile == 'Functionally weak':
        recommendation += "Comprehensive approach needed to address fundamental movement patterns. "
        recommendation += "Focus on basic movement quality before progressing to higher loads. "
        recommendation += "Recommend 3x weekly corrective exercise sessions with regular FMS reassessment. "
    
    elif profile == 'Strength-deficient':
        recommendation += "Progressive resistance training required to address strength deficits. "
        recommendation += "Recommend compound movements 2-3x weekly with gradual progression of load. "
        
        # Check specific strength areas
        if athlete['flag_strength'] == 1:
            # Identify specific weakness areas
            strength_cols = ['HAbd_L', 'HAbd_P', 'KE_L', 'KE_P', 'KF_L', 'KF_P', 'AP_L', 'AP_P']
            weakness_areas = []
            
            for col in strength_cols:
                if col in df.columns and zscore(df[col])[athlete_index] < -0.5:
                    area = col.split('_')[0]
                    side = "left" if col.endswith('_L') else "right"
                    weakness_areas.append(f"{area} ({side})")
            
            if weakness_areas:
                recommendation += f"Pay special attention to: {', '.join(weakness_areas[:3])}. "
    
    elif profile == 'Stability-deficient':
        recommendation += "Focus on neuromuscular control and proprioceptive exercises. "
        recommendation += "Recommend balance training and controlled mobility work 3-4x weekly. "
        
        # Check specific stability areas
        if athlete['flag_ybt'] == 1:
            # Identify specific stability issues
            ybt_cols = ['YBT_ANT_L', 'YBT_ANT_P', 'YBT_PM_L', 'YBT_PM_P', 'YBT_PL_L', 'YBT_PL_P']
            stability_issues = []
            
            for col in ybt_cols:
                if col in df.columns and zscore(df[col])[athlete_index] < -0.5:
                    direction = col.split('_')[1]
                    side = "left" if col.endswith('_L') else "right"
                    stability_issues.append(f"{direction} direction ({side})")
            
            if stability_issues:
                recommendation += f"Stability deficits particularly noted in: {', '.join(stability_issues[:3])}. "
    
    elif profile == 'No clear dysfunction':
        recommendation += "Functional profile is within normal ranges. "
        recommendation += "Continue with current training regime with focus on maintenance and progression. "
    
    # Flag-specific recommendations
    if flags == "Red flag":
        recommendation += "IMMEDIATE ACTION REQUIRED: Multiple high-risk factors detected. "
        recommendation += "Consider reduced training volume until deficits are addressed. "
        recommendation += "Weekly monitoring recommended during corrective phase."
    
    elif flags == "Yellow flag":
        recommendation += "CAUTION ADVISED: Moderate risk factors present. "
        recommendation += "Implement corrective strategies alongside regular training. "
        recommendation += "Bi-weekly reassessment recommended."
    
    elif flags == "Green flag":
        recommendation += "LOW RISK PROFILE: Continue regular training with periodic assessment. "
        recommendation += "Maintain focus on overall athletic development."
    
    # Check for asymmetries
    left_right_diffs = []
    for label, (left_col, right_col) in st.session_state.limb_columns.items():
        if left_col in df.columns and right_col in df.columns:
            left_val = athlete[left_col]
            right_val = athlete[right_col]
            if abs(left_val - right_val) / ((left_val + right_val) / 2) > 0.15:  # 15% asymmetry
                left_right_diffs.append(label)
    
    if left_right_diffs:
        recommendation += f"\n\nSIGNIFICANT ASYMMETRY DETECTED in: {', '.join(left_right_diffs)}. "
        recommendation += "Recommend targeted unilateral exercises to address imbalances."
    
    return recommendation

def generate_html_report(df, athlete_index):
    """Generate HTML report for an athlete"""
    if df is None or athlete_index >= len(df):
        return None
    
    athlete = df.iloc[athlete_index]
    
    # Generate radar chart and convert to base64
    fig = generate_radar_chart(df, athlete_index)
    if fig:
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100)
        plt.close(fig)
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
    else:
        img_str = ""
    
    recommendation = generate_recommendation(df, athlete_index)
    
    # Try to get athlete name
    athlete_names = get_athlete_names(df)
    athlete_name = athlete_names[athlete_index] if athlete_index < len(athlete_names) else f"Athlete #{athlete_index}"
    
    # Flag colors
    flag_colors = {
        "Red flag": "#ffcccc",
        "Yellow flag": "#fff3cd",
        "Green flag": "#d4edda"
    }
    
    # Classification colors
    classification_colors = {
        'Functionally weak': "#ffcccc",
        'Strength-deficient': "#fff3cd",
        'Stability-deficient': "#fff3cd",
        'No clear dysfunction': "#d4edda"
    }
    
    # HTML template
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Functional Report: {{ athlete_name }}</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                line-height: 1.6;
                margin: 0;
                padding: 20px;
                color: #333;
            }
            .header {
                text-align: center;
                margin-bottom: 30px;
                padding-bottom: 10px;
                border-bottom: 2px solid #ddd;
            }
            .container {
                max-width: 1000px;
                margin: 0 auto;
            }
            .profile-section {
                display: flex;
                margin-bottom: 30px;
            }
            .profile-info {
                flex: 1;
                padding: 20px;
                background-color: #f8f9fa;
                border-radius: 5px;
                margin-right: 20px;
            }
            .chart-section {
                flex: 1;
                text-align: center;
            }
            .recommendation-section {
                padding: 20px;
                border-radius: 5px;
                background-color: {{ flag_color }};
                margin-top: 20px;
                margin-bottom: 20px;
            }
            table {
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
            }
            th, td {
                padding: 12px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }
            th {
                background-color: #f2f2f2;
            }
            .footer {
                margin-top: 50px;
                text-align: center;
                color: #777;
                font-size: 12px;
                border-top: 1px solid #ddd;
                padding-top: 10px;
            }
            .metrics-card {
                display: flex;
                justify-content: space-between;
                margin-top: 20px;
            }
            .metric {
                text-align: center;
                padding: 10px;
                background-color: #f8f9fa;
                border-radius: 5px;
                flex: 1;
                margin: 0 5px;
            }
            .metric.deficit {
                background-color: #ffcccc;
            }
            .metric.caution {
                background-color: #fff3cd;
            }
            .metric.good {
                background-color: #d4edda;
            }
            .metric-value {
                font-size: 24px;
                font-weight: bold;
                margin: 10px 0;
            }
            .metric-label {
                font-size: 14px;
                color: #666;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Athlete Functional Assessment Report</h1>
                <h2>{{ athlete_name }}</h2>
                <p>Report Date: {{ today }}</p>
            </div>
            
            <div class="profile-section">
                <div class="profile-info">
                    <h3>Athlete Profile</h3>
                    <table>
                        <tr>
                            <th>Metric</th>
                            <th>Value</th>
                            <th>Z-Score</th>
                        </tr>
                        <tr>
                            <td>FMS Total</td>
                            <td>{{ athlete.FMS_TOTAL }}</td>
                            <td>{{ fms_percentile }}%ile</td>
                        </tr>
                        <tr>
                            <td>Mean Strength</td>
                            <td>{{ athlete.Mean_Strength|round(2) }}</td>
                            <td>{{ athlete.Mean_Strength_Z|round(2) }}</td>
                        </tr>
                        <tr>
                            <td>Mean YBT Score</td>
                            <td>{{ athlete.Mean_YBT|round(2) }}</td>
                            <td>{{ athlete.Mean_YBT_Z|round(2) }}</td>
                        </tr>
                        <tr>
                            <td>BMI</td>
                            <td>{{ athlete.BMI|round(2) }}</td>
                            <td>{{ athlete.BMI_Z|round(2) }}</td>
                        </tr>
                    </table>
                    
                    <div class="metrics-card">
                        <div class="metric {{ 'deficit' if athlete.flag_fms == 1 else 'good' }}">
                            <div class="metric-label">FMS Status</div>
                            <div class="metric-value">{{ athlete.FMS_category }}</div>
                        </div>
                        <div class="metric {{ 'deficit' if athlete.flag_strength == 1 else 'good' }}">
                            <div class="metric-label">Strength Status</div>
                            <div class="metric-value">{{ "Deficit" if athlete.flag_strength == 1 else "Normal" }}</div>
                        </div>
                        <div class="metric {{ 'deficit' if athlete.flag_ybt == 1 else 'good' }}">
                            <div class="metric-label">Stability Status</div>
                            <div class="metric-value">{{ "Deficit" if athlete.flag_ybt == 1 else "Normal" }}</div>
                        </div>
                    </div>
                    
                    <h3 style="margin-top: 30px;">Functional Classification</h3>
                    <div class="metric" style="background-color: {{ classification_color }}; text-align: center; padding: 15px; margin-top: 10px;">
                        <div class="metric-value" style="font-size: 28px;">{{ athlete.Deficit_Profile }}</div>
                        <div class="metric-label" style="font-size: 16px;">Risk Level: {{ athlete.Flag_Category }}</div>
                    </div>
                </div>
                
                <div class="chart-section">
                    <h3>Z-score Functional Profile: Left vs Right Limb</h3>
                    <img src="data:image/png;base64,{{ radar_chart }}" alt="Radar Chart" style="max-width: 100%; height: auto;">
                </div>
            </div>
            
            <div class="recommendation-section">
                <h3>Personalized Recommendations</h3>
                <p style="white-space: pre-line;">{{ recommendation }}</p>
            </div>
            
            <div class="footer">
                <p>This report is generated automatically based on functional assessment data. It should be interpreted by qualified sports medicine professionals.</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    # Create template and render HTML
    template = Template(html_template)
    html = template.render(
        athlete_name=athlete_name,
        athlete=athlete,
        radar_chart=img_str,
        recommendation=recommendation,
        today=datetime.now().strftime("%Y-%m-%d"),
        fms_percentile=round(athlete['FMS_percentile']),
        flag_color=flag_colors.get(athlete['Flag_Category'], "#ffffff"),
        classification_color=classification_colors.get(athlete['Deficit_Profile'], "#ffffff")
    )
    
    return html

# Main application logic
if uploaded_file is not None:
    try:
        with st.spinner("Loading and processing data..."):
            # Load data
            df = pd.read_excel(uploaded_file)
            
            # Process data
            processed_df = process_data(df)
            
            if processed_df is not None:
                st.session_state.df = processed_df
                st.session_state.processed = True
                st.success("Data processed successfully!")
            else:
                st.error("Error processing data. Please check your Excel file format.")
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")

# Display data analysis if data is processed
if st.session_state.processed and st.session_state.df is not None:
    st.header("Athlete Analysis")
    
    # Get athlete names
    athlete_names = get_athlete_names(st.session_state.df)
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Dashboard", "Data Preview", "Reports"])
    
    with tab1:
        # Dashboard layout
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Athlete selection
            selected_athlete = st.selectbox("Select Athlete:", athlete_names)
            athlete_index = athlete_names.index(selected_athlete)
            
            # Profile info
            athlete = st.session_state.df.iloc[athlete_index]
            
            st.subheader("Functional Profile")
            
            # Style the profile box with appropriate color
            profile_colors = {
                'Functionally weak': "#ffcccc",
                'Strength-deficient': "#fff3cd",
                'Stability-deficient': "#fff3cd",
                'No clear dysfunction': "#d4edda"
            }
            profile_color = profile_colors.get(athlete['Deficit_Profile'], "#ffffff")
            
            st.markdown(
                f"""
                <div style="background-color: {profile_color}; padding: 10px; border-radius: 5px; margin-bottom: 20px;">
                    <h3 style="margin: 0; text-align: center;">{athlete['Deficit_Profile']}</h3>
                    <p style="margin: 5px 0 0 0; text-align: center;">Risk Level: {athlete['Flag_Category']}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            # Metrics
            col1a, col1b, col1c = st.columns(3)
            
            with col1a:
                st.metric("FMS", f"{athlete['FMS_TOTAL']:.1f}", f"{athlete['FMS_percentile']:.0f}%ile")
            
            with col1b:
                st.metric("Strength", f"{athlete['Mean_Strength']:.2f}", f"{athlete['Mean_Strength_Z']:.2f} z")
            
            with col1c:
                st.metric("Stability", f"{athlete['Mean_YBT']:.2f}", f"{athlete['Mean_YBT_Z']:.2f} z")
            
            # Recommendations
            st.subheader("Recommendations")
            recommendation = generate_recommendation(st.session_state.df, athlete_index)
            
            # Display recommendation with colored background based on risk level
            flag_colors = {
                "Red flag": "#ffcccc",
                "Yellow flag": "#fff3cd",
                "Green flag": "#d4edda"
            }
            flag_color = flag_colors.get(athlete['Flag_Category'], "#ffffff")
            
            st.markdown(
                f"""
                <div style="background-color: {flag_color}; padding: 15px; border-radius: 5px; margin-top: 10px;">
                    <p style="margin: 0;">{recommendation}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            # Check for asymmetries
            st.subheader("Asymmetry Analysis")
            
            asymmetry_data = []
            for label, (left_col, right_col) in st.session_state.limb_columns.items():
                if left_col in st.session_state.df.columns and right_col in st.session_state.df.columns:
                    left_val = athlete[left_col]
                    right_val = athlete[right_col]
                    diff_pct = abs(left_val - right_val) / ((left_val + right_val) / 2) * 100
                    stronger = "Left" if left_val > right_val else "Right"
                    
                    asymmetry_data.append({
                        "Measurement": label,
                        "Left": f"{left_val:.2f}",
                        "Right": f"{right_val:.2f}",
                        "Difference (%)": f"{diff_pct:.1f}%",
                        "Stronger Side": stronger
                    })
            
            asymmetry_df = pd.DataFrame(asymmetry_data)
            st.dataframe(asymmetry_df, use_container_width=True)
        
        with col2:
            # Radar chart
            st.subheader("Functional Profile Radar Chart")
            radar_fig = generate_radar_chart(st.session_state.df, athlete_index)
            if radar_fig:
                st.pyplot(radar_fig)
    
    with tab2:
        # Data preview
        st.subheader("Processed Data Preview")
        st.dataframe(st.session_state.df)
        
        # Distribution of profiles
        st.subheader("Functional Profile Distribution")
        profile_counts = st.session_state.df['Deficit_Profile'].value_counts()
        st.bar_chart(profile_counts)
    
    with tab3:
        # Generate reports
        st.subheader("Generate Reports")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Individual report
            st.markdown("### Individual Report")
            report_athlete = st.selectbox("Select athlete for report:", athlete_names)
            report_index = athlete_names.index(report_athlete)
            
            if st.button("Generate Individual Report"):
                with st.spinner("Generating report..."):
                    html_report = generate_html_report(st.session_state.df, report_index)
                    if html_report:
                        # Create download button for HTML
                        safe_name = report_athlete.replace(" ", "_").replace("#", "")
                        st.download_button(
                            label="Download HTML Report",
                            data=html_report,
                            file_name=f"{safe_name}_report.html",
                            mime="text/html"
                        )
                        
                        # Show preview
                        st.markdown("### Report Preview")
                        st.components.v1.html(html_report, height=600, scrolling=True)
                    else:
                        st.error("Error generating report")
        
        with col2:
            # Batch reports
            st.markdown("### Batch Reports")
            st.write("Generate reports for multiple athletes at once.")
            
            if st.button("Generate All Reports"):
                # Create a ZIP file with all reports
                import zipfile
                from io import BytesIO
                
                zip_buffer = BytesIO()
                with zipfile.ZipFile(zip_buffer, "w") as zip_file:
                    for i, name in enumerate(athlete_names):
                        with st.spinner(f"Generating report for {name}..."):
                            html_report = generate_html_report(st.session_state.df, i)
                            if html_report:
                                safe_name = name.replace(" ", "_").replace("#", "")
                                zip_file.writestr(f"{safe_name}_report.html", html_report)
                
                zip_buffer.seek(0)
                st.download_button(
                    label="Download All Reports (ZIP)",
                    data=zip_buffer,
                    file_name="athlete_reports.zip",
                    mime="application/zip"
                )