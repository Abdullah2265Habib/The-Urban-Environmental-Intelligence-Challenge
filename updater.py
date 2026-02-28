import re

with open('streamlit_dashboard.py', 'r', encoding='utf-8') as f:
    content = f.read()

new_functions = """
def prepare_data_for_heatmap(df):
    pm25_df = df[df['parameter'] == 'pm25'].copy()
    time_col = 'datetimeLocal' if 'datetimeLocal' in pm25_df.columns else 'datetimeUtc'
    pm25_df['datetime'] = pd.to_datetime(pm25_df[time_col], utc=True).dt.tz_convert(None)
    pm25_df['hour'] = pm25_df['datetime'].dt.hour
    pm25_df['day_of_year'] = pm25_df['datetime'].dt.dayofyear
    pm25_df['month'] = pm25_df['datetime'].dt.month
    pm25_df['location_label'] = pm25_df['location_id'].astype(str) + " - " + pm25_df['location_name'].astype(str).str[:15]
    return pm25_df

def create_high_density_heatmap(df, time_resolution='hour'):
    if time_resolution == 'hour':
        heatmap_data = df.groupby(['location_label', 'hour'])['value'].mean().reset_index()
        pivot_df = heatmap_data.pivot(index='location_label', columns='hour', values='value')
        x_label = "Hour of Day (0-23)"
        title = "Daily Periodic Signature (Average PM2.5 by Hour)"
    elif time_resolution == 'day':
        heatmap_data = df.groupby(['location_label', 'day_of_year'])['value'].mean().reset_index()
        pivot_df = heatmap_data.pivot(index='location_label', columns='day_of_year', values='value')
        x_label = "Day of Year (1-365)"
        title = "Seasonal Periodic Signature (Average PM2.5 by Day)"
    else:
        raise ValueError("Invalid time resolution")
        
    if len(pivot_df) > 100:
        pivot_df = pivot_df.head(100)
    pivot_df = pivot_df.sort_index(ascending=False) 

    fig = go.Figure(data=go.Heatmap(
        z=pivot_df.values,
        x=pivot_df.columns,
        y=pivot_df.index,
        colorscale=[
            [0.0, 'lightgreen'],    # Excellent
            [0.2, 'green'],         # Good (0-15)
            [0.35, 'yellow'],       # Moderate (15-35)
            [0.3501, 'orange'],     # Health Hazard Limit threshold (35)
            [0.55, 'red'],          # Unhealthy
            [0.8, 'purple'],        # Very Unhealthy
            [1.0, 'maroon']         # Hazardous
        ],
        zmin=0,
        zmax=100,
        colorbar=dict(
            title="PM2.5 (µg/m³)",
            tickvals=[0, 15, 35, 55, 75],
            ticktext=['0', '15', '35 (Hazard)', '55', '75+']
        ),
        hovertemplate='Location: %{y}<br>Time: %{x}<br>PM2.5: %{z:.1f} µg/m³<extra></extra>'
    ))

    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title="Sensor Locations",
        height=800,  
        width=1000,
        yaxis=dict(
            title_standoff=0,
            tickfont=dict(size=8),
            autorange='reversed'
        ),
        template='plotly_white',
        margin=dict(l=150, r=20, t=60, b=40)
    )
    
    fig.add_annotation(
        text="Target: Identify values > 35 µg/m³ (Orange/Red)",
        xref="paper", yref="paper",
        x=1.0, y=1.05,
        showarrow=False,
        font=dict(size=12, color="red")
    )

    return fig

def run_task_2(df):
    st.header("🕒 Task 2: High-Density Temporal Analysis")
    st.markdown('''
    **Objective:** Identify "Health Threshold Violations" (PM2.5 > 35 µg/m³) across all 100 sensors simultaneously
    using a high-density temporal visualization to prevent "spaghetti chart" clutter.
    ''')
    
    with st.spinner("Processing temporal data..."):
        pm25_df = prepare_data_for_heatmap(df)
        
    if pm25_df.empty:
        st.error("No PM2.5 data found in the dataset.")
        return
        
    st.markdown("---")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        time_res = st.radio(
            "Select Time Resolution to find Periodic Signatures:",
            options=['Daily (Hour of Day)', 'Seasonal (Day of Year)'],
            help="Switch between daily cycles (traffic patterns) and seasonal/monthly shifts.",
            horizontal=True
        )
        resolution_key = 'hour' if 'Daily' in time_res else 'day'
        
    with col2:
        st.markdown('''
        **Color Legend**
        - 🟢 **< 15**: Good
        - 🟡 **15 - 35**: Moderate
        - 🟠 **35 - 55**: Unhealthy (Violation Target)
        - 🔴 **> 55**: Hazardous
        ''')
        
    fig_map = create_high_density_heatmap(pm25_df, time_resolution=resolution_key)
    st.plotly_chart(fig_map, use_container_width=True)
        
    st.markdown("---")
    
    st.header("💡 Analysis & Findings")
    st.markdown('''
    ### Determining the Periodic Signature
    
    By switching between the Daily and Seasonal views above, we avoid the overplotting of 100 line charts and clearly see the pollution patterns:
    
    1. **Daily Signatures (24-hour cycle):** Look at the 'Daily' resolution. If you observe intense vertical bands of color (Orange/Red > 35 µg/m³) consistently appearing around specific hours (e.g., 07:00-09:00 and 17:00-19:00), this strongly indicates **traffic-driven periodic signatures** typical of rush hours.
    2. **Seasonal Signatures (Monthly/Yearly):** Look at the 'Seasonal' resolution. If solid blocks of intense pollution span multiple consecutive days or weeks (e.g., dense red bands clustered in winter months: Day 1-60 and Day 300-365), this suggests **weather-driven or seasonal periodic signatures** such as increased heating demand or atmospheric inversions.
    
    *Constraint Checked: Avoided "Spaghetti Chart" clutter by using a Heatmap with High-Density 2D layout. Minimized scale distortion by enforcing a custom color gradient centered around the Mayor's critical threshold of 35 µg/m³.*
    ''')
"""

old_main_body = re.search(r'def main\(\):(.*?)(?=if __name__ == "__main__":)', content, flags=re.DOTALL)
if old_main_body:
    old_main_content = old_main_body.group(1)
    
    # We replace the old main entirely with our new structure
    prefix = content[:content.find('def main():')]
    suffix = content[content.find('if __name__ == "__main__":'):]
    
    # Modify old main to fit in run_task_1 (remove the initial page load logic that we will move to the new main)
    old_main_content = re.sub(r'    # Title and Description.*?(?=    # Sidebar Configuration)', '', old_main_content, flags=re.DOTALL)
    
    run_task_1_content = "def run_task_1(df):" + old_main_content
    
    final_main = """
def main():
    st.title("🌍 Urban Environmental Intelligence Challenge")
    st.markdown("### Interactive Environmental Diagnostics Engine")
    
    st.info("📊 Loading environmental data from 100 sensors... (This may take a moment)")
    df = load_data()
    
    tab1, tab2 = st.tabs(["Task 1: Dimensionality Reduction", "Task 2: Temporal Analysis"])
    
    with tab1:
        run_task_1(df)
        
    with tab2:
        run_task_2(df)

"""
    new_content = prefix + new_functions + "\n" + run_task_1_content + "\n" + final_main + "\n" + suffix
    
    with open('streamlit_dashboard.py', 'w', encoding='utf-8') as f:
        f.write(new_content)
    print("Dashboard updated successfully!")
else:
    print('Failed to parse old main')
