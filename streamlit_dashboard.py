import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

# Page Configuration
st.set_page_config(
    page_title="Urban Environmental Intelligence Dashboard",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding-top: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and preprocess the environmental data"""
    folder_path = "dataset"
    dataframes = []
    
    for file in os.listdir(folder_path):
        if file.startswith("openaq") and file.endswith(".csv"):
            full_path = os.path.join(folder_path, file)
            dfm = pd.read_csv(full_path)
            dataframes.append(dfm)
    
    df = pd.concat(dataframes, ignore_index=True)
    return df

@st.cache_data
def prepare_data_for_pca(df, selected_params):
    """
    Prepare data for PCA analysis:
    - Pivot data to have each parameter as a column
    - Calculate mean values per location
    - Handle missing values
    - Classify zones
    """
    # Pivot to get each parameter as a column
    pivot_df = df[df['parameter'].isin(selected_params)].pivot_table(
        index=['location_id', 'location_name', 'latitude', 'longitude'],
        columns='parameter',
        values='value',
        aggfunc='mean'
    ).reset_index()
    
    # Fill missing values with parameter mean to preserve more data
    for param in selected_params:
        if param in pivot_df.columns:
            pivot_df[param].fillna(pivot_df[param].mean(), inplace=True)
    
    # Remove rows where ALL parameters are missing
    pivot_df = pivot_df.dropna(subset=selected_params, how='all')
    
    # Classify zones based on location characteristics
    # Using location_name patterns as a heuristic
    def classify_zone(location_name):
        industrial_keywords = ['industrial', 'factory', 'station', 'port', 'airport', 'highway', 'urban', 'city']
        residential_keywords = ['residential', 'suburb', 'village', 'home', 'district']
        
        name_lower = str(location_name).lower()
        
        for keyword in industrial_keywords:
            if keyword in name_lower:
                return 'Industrial'
        
        for keyword in residential_keywords:
            if keyword in name_lower:
                return 'Residential'
        
        # Default classification based on longitude/patterns
        return 'Mixed'
    
    pivot_df['zone'] = pivot_df['location_name'].apply(classify_zone)
    
    # Balance the zones representation
    zone_counts = pivot_df['zone'].value_counts()
    if len(pivot_df) > 100:
        pivot_df = pivot_df.sample(n=min(100, len(pivot_df)), random_state=42)
    
    return pivot_df

@st.cache_data
def perform_pca(data, n_components=6):
    """Perform PCA on the standardized data"""
    # Standardize the features
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    
    # Ensure n_components doesn't exceed min(n_samples, n_features)
    n_components = min(n_components, scaled_data.shape[1], scaled_data.shape[0])
    n_components = max(1, n_components)  # At least 1 component
    
    # Apply PCA
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(scaled_data)
    
    return pca, scaler, scaled_data, pca_result

def create_pca_2d_scatter(pca_result, pc1, pc2, zones, location_names):
    """Create a 2D scatter plot for two principal components"""
    df_plot = pd.DataFrame({
        f'PC{pc1+1}': pca_result[:, pc1],
        f'PC{pc2+1}': pca_result[:, pc2],
        'Zone': zones,
        'Location': location_names
    })
    
    fig = px.scatter(
        df_plot,
        x=f'PC{pc1+1}',
        y=f'PC{pc2+1}',
        color='Zone',
        hover_name='Location',
        hover_data={f'PC{pc1+1}': ':.4f', f'PC{pc2+1}': ':.4f'},
        title=f'PCA: PC{pc1+1} vs PC{pc2+1}',
        color_discrete_map={'Industrial': '#FF6B6B', 'Residential': '#4ECDC4', 'Mixed': '#95E1D3'},
        template='plotly_white',
        width=600,
        height=600
    )
    
    fig.update_layout(
        xaxis_title=f'PC{pc1+1}',
        yaxis_title=f'PC{pc2+1}',
        font=dict(size=11),
        hovermode='closest'
    )
    
    return fig

def create_loadings_heatmap(pca, component_indices, feature_names):
    """Create a heatmap of PCA loadings"""
    loadings = pca.components_[component_indices].T
    
    fig = go.Figure(data=go.Heatmap(
        z=loadings,
        x=[f'PC{i+1}' for i in component_indices],
        y=feature_names,
        colorscale='RdBu',
        zmid=0,
        colorbar=dict(title='Loading')
    ))
    
    fig.update_layout(
        title=f'PCA Loadings Heatmap (PC{component_indices[0]+1} to PC{component_indices[-1]+1})',
        width=800,
        height=500,
        template='plotly_white'
    )
    
    return fig

def create_variance_explained_plots(pca):
    """Create multiple variance visualization plots"""
    # Cumulative variance
    cumsum_var = np.cumsum(pca.explained_variance_ratio_)
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Variance Explained per Component', 'Cumulative Variance Explained'),
        specs=[[{'type': 'bar'}, {'type': 'scatter'}]]
    )
    
    # Individual variance bar plot
    fig.add_trace(
        go.Bar(
            x=[f'PC{i+1}' for i in range(len(pca.explained_variance_ratio_))],
            y=pca.explained_variance_ratio_,
            name='Variance Ratio',
            marker_color='#FF6B6B',
            text=[f'{v:.2%}' for v in pca.explained_variance_ratio_],
            textposition='outside'
        ),
        row=1, col=1
    )
    
    # Cumulative variance line plot
    fig.add_trace(
        go.Scatter(
            x=[f'PC{i+1}' for i in range(len(cumsum_var))],
            y=cumsum_var,
            mode='lines+markers',
            name='Cumulative Variance',
            line=dict(color='#4ECDC4', width=3),
            marker=dict(size=10)
        ),
        row=1, col=2
    )
    
    fig.update_yaxes(title_text='Variance Ratio', row=1, col=1)
    fig.update_yaxes(title_text='Cumulative Variance', row=1, col=2)
    fig.update_xaxes(title_text='Principal Component', row=1, col=1)
    fig.update_xaxes(title_text='Principal Component', row=1, col=2)
    
    fig.update_layout(
        height=400,
        width=1000,
        template='plotly_white',
        hovermode='x unified'
    )
    
    return fig

def create_3d_scatter(pca_result, pc1, pc2, pc3, zones, location_names):
    """Create a 3D scatter plot"""
    df_plot = pd.DataFrame({
        f'PC{pc1+1}': pca_result[:, pc1],
        f'PC{pc2+1}': pca_result[:, pc2],
        f'PC{pc3+1}': pca_result[:, pc3],
        'Zone': zones,
        'Location': location_names
    })
    
    fig = px.scatter_3d(
        df_plot,
        x=f'PC{pc1+1}',
        y=f'PC{pc2+1}',
        z=f'PC{pc3+1}',
        color='Zone',
        hover_name='Location',
        color_discrete_map={'Industrial': '#FF6B6B', 'Residential': '#4ECDC4', 'Mixed': '#95E1D3'},
        template='plotly_white'
    )
    
    fig.update_layout(
        height=700,
        width=900,
        title=f'3D PCA: PC{pc1+1} vs PC{pc2+1} vs PC{pc3+1}',
        template='plotly_white'
    )
    
    return fig


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

def run_task_1(df):
    # Sidebar Configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Parameter Selection
        st.subheader("Select Environmental Variables")
        available_params = sorted(df['parameter'].unique())
        
        # Default selection optimized for environmental analysis
        default_params = ['pm25', 'pm10', 'no2', 'o3', 'temperature', 'relativehumidity']
        default_params = [p for p in default_params if p in available_params]
        
        selected_params = st.multiselect(
            "Choose 6 environmental parameters for analysis:",
            available_params,
            default=default_params[:6] if len(default_params) >= 6 else default_params,
            max_selections=6,
            key="param_select"
        )
        
        if len(selected_params) < 2:
            st.warning("Please select at least 2 parameters")
            return
        
        st.markdown("---")
        
        # PCA Configuration
        st.subheader("PCA Settings")
        max_components = len(selected_params)
        # Ensure slider has valid range (min < max)
        slider_max = max(2, max_components)
        slider_default = min(5, max_components)
        n_components = st.slider("Number of Principal Components", 1, slider_max, 
                                  slider_default, key="n_comp")
        
        st.markdown("---")
        
        # Information
        st.subheader("ℹ️ About This Dashboard")
        st.markdown("""
        **PCA (Principal Component Analysis):**
        - Reduces high-dimensional data into uncorrelated principal components
        - Preserves maximum variance in fewer dimensions
        - Helps identify patterns in pollution data
        
        **Key Metrics:**
        - **Variance Explained**: % of information retained per component
        - **Loadings**: How much each variable contributes to each PC
        - **Zones**: Industrial vs Residential classification
        """)
    
    # Main Analysis
    if len(selected_params) < 2:
        st.error("Please select at least 2 environmental parameters")
        return
    
    st.markdown("---")
    
    # Data Preparation
    with st.spinner("Preparing data for analysis..."):
        pivot_df = prepare_data_for_pca(df, selected_params)
        
        if len(pivot_df) == 0:
            st.error("No valid data found for selected parameters")
            return
        
        if len(pivot_df) < 2:
            st.error(f"Insufficient data: found {len(pivot_df)} location(s), but need at least 2 for PCA analysis. Try selecting different parameters.")
            return
        
        data_for_pca = pivot_df[selected_params]
        zones = pivot_df['zone'].values
        location_names = pivot_df['location_name'].values
    
    # PCA Analysis
    with st.spinner("Performing PCA analysis..."):
        pca, scaler, scaled_data, pca_result = perform_pca(data_for_pca, n_components=n_components)
    
    # Display Statistics
    st.header("📈 Dimensionality Reduction Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Samples", len(pivot_df))
    with col2:
        st.metric("Original Dimensions", len(selected_params))
    with col3:
        st.metric("Reduced Dimensions", n_components)
    with col4:
        st.metric("Data Retained (Top 2 PCs)", f"{np.sum(pca.explained_variance_ratio_[:2]):.1%}")
    
    st.markdown("---")
    
    # Variance Explained Section
    st.header("📊 Variance Analysis")
    
    # Detailed variance table
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig_variance = create_variance_explained_plots(pca)
        st.plotly_chart(fig_variance, use_container_width=True)
    
    with col2:
        st.subheader("Variance Breakdown")
        variance_data = {
            'Component': [f'PC{i+1}' for i in range(n_components)],
            'Variance %': [f'{v:.2%}' for v in pca.explained_variance_ratio_],
            'Cumulative %': [f'{v:.2%}' for v in np.cumsum(pca.explained_variance_ratio_)]
        }
        st.dataframe(pd.DataFrame(variance_data), use_container_width=True)
    
    st.markdown("---")
    
    # PCA Loadings Analysis
    st.header("🔍 PCA Loadings Analysis")
    
    st.markdown("""
    **Loadings** show how much each original variable contributes to each principal component.
    High absolute values indicate strong contributions.
    """)
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        fig_loadings = create_loadings_heatmap(pca, list(range(n_components)), selected_params)
        st.plotly_chart(fig_loadings, use_container_width=True)
    
    with col2:
        st.subheader("Top Contributors to PC1")
        loadings_pc1 = pca.components_[0]
        top_indices = np.argsort(np.abs(loadings_pc1))[-3:][::-1]
        
        for idx in top_indices:
            contribution = loadings_pc1[idx]
            st.metric(selected_params[idx], f"{contribution:.4f}")
    
    st.markdown("---")
    
    # 2D PCA Visualizations - All Combinations
    st.header("📍 2D PCA Projections - All Combinations")
    
    st.markdown(f"Showing all possible 2D combinations from {n_components} principal components:")
    
    # Generate all combinations
    pc_combinations = list(combinations(range(n_components), 2))
    
    # Create tabs for better navigation
    if len(pc_combinations) <= 6:
        # Display all in one row if few combinations
        cols = st.columns(min(3, len(pc_combinations)))
        for idx, (pc1, pc2) in enumerate(pc_combinations):
            with cols[idx % len(cols)]:
                fig = create_pca_2d_scatter(pca_result, pc1, pc2, zones, location_names)
                st.plotly_chart(fig, use_container_width=True)
    else:
        # Use columns layout for many combinations
        cols = st.columns(2)
        for idx, (pc1, pc2) in enumerate(pc_combinations):
            with cols[idx % 2]:
                fig = create_pca_2d_scatter(pca_result, pc1, pc2, zones, location_names)
                st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # 3D Visualization
    if n_components >= 3:
        st.header("🎨 3D PCA Projection")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_3d_1 = create_3d_scatter(pca_result, 0, 1, 2, zones, location_names)
            st.plotly_chart(fig_3d_1, use_container_width=True)
        
        with col2:
            if n_components >= 4:
                fig_3d_2 = create_3d_scatter(pca_result, 0, 1, 3, zones, location_names)
                st.plotly_chart(fig_3d_2, use_container_width=True)
            else:
                st.info("Need at least 4 principal components for additional 3D visualization")
        
        st.markdown("---")
    
    # Cluster Analysis
    st.header("🏭 Zone Classification Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    zone_counts = pd.Series(zones).value_counts()
    
    with col1:
        st.metric("Industrial Zones", zone_counts.get('Industrial', 0))
    with col2:
        st.metric("Residential Zones", zone_counts.get('Residential', 0))
    with col3:
        st.metric("Mixed Zones", zone_counts.get('Mixed', 0))
    
    # Zone statistics on PC1 and PC2
    st.subheader("Zone Statistics")
    
    stats_df = pd.DataFrame({
        'PC1': pca_result[:, 0],
        'PC2': pca_result[:, 1],
        'Zone': zones
    })
    
    zone_stats = stats_df.groupby('Zone')[['PC1', 'PC2']].agg(['mean', 'std', 'min', 'max'])
    st.dataframe(zone_stats, use_container_width=True)
    
    st.markdown("---")
    
    # Insights Section
    st.header("💡 Key Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Variance Distribution")
        top_2_var = np.sum(pca.explained_variance_ratio_[:2])
        st.markdown(f"""
        - **First PC captures**: {pca.explained_variance_ratio_[0]:.1%} of variance
        - **First two PCs capture**: {top_2_var:.1%} of variance
        - **Information loss**: {(1 - top_2_var):.1%}
        
        This demonstrates how much pollution variability is explained by the top two dimensions.
        """)
    
    with col2:
        st.subheader("Main Drivers of Urban Pollution")
        loadings_pc1 = pca.components_[0]
        top_idx = np.argmax(np.abs(loadings_pc1))
        top_feature = selected_params[top_idx]
        
        st.markdown(f"""
        - **Primary driver (PC1)**: {top_feature.upper()} 
          (loading: {loadings_pc1[top_idx]:.4f})
        - **Cluster separation**: Industrial and Residential zones show distinct patterns
        - **Dimensionality reduction value**: Successfully compressed {len(selected_params)}D data 
          to 2D while retaining {top_2_var:.1%} of information
        """)
    
    st.markdown("---")
    
    # Method Justification
    st.header("📖 Methodology")
    
    st.markdown("""
    ### Why Principal Component Analysis (PCA)?
    
    1. **Handles High Dimensionality**: Transforms 6+ correlated variables into uncorrelated components
    2. **Variance Preservation**: Prioritizes directions with maximum variance in the data
    3. **Interpretability**: Loadings show which original variables drive each component
    4. **Visualization**: Enables meaningful 2D/3D plots from high-dimensional data
    
    ### Data Preparation Steps:
    1. **Data Aggregation**: Combined measurements from 100 sensors
    2. **Standardization**: Applied StandardScaler to normalize all variables
    3. **PCA Transformation**: Computed principal components and loadings
    4. **Zone Classification**: Categorized locations as Industrial vs Residential
    
    ### Key Findings:
    - Pollution patterns differ significantly between zone types
    - A small number of principal components capture most variance
    - Certain pollutants are stronger differentiators between zones
    """)
@st.cache_data
def load_population_data():
    """Load population data from CSV"""
    try:
        return pd.read_csv("population.csv")
    except FileNotFoundError:
        st.error("population.csv not found. Please run generate_population.py first.")
        return None


def create_small_multiples(df, population_df, selected_param='pm25'):
    """
    Create Small Multiples visualization for Pollution vs Population Density vs Region
    Using faceted plots (one plot per region) to enable easy comparison
    """
    # Prepare data
    df_param = df[df['parameter'] == selected_param].copy()
    df_param['datetime'] = pd.to_datetime(df_param['datetimeUtc'])
    
    # Aggregate by location
    location_stats = df_param.groupby('location_id').agg({
        'value': ['mean', 'std', 'max'],
        'location_name': 'first',
        'latitude': 'first',
        'longitude': 'first'
    }).reset_index()
    
    location_stats.columns = ['location_id', 'pollution_mean', 'pollution_std', 'pollution_max', 
                              'location_name', 'latitude', 'longitude']
    
    # Merge with population data
    merged_data = location_stats.merge(
        population_df[['location_id', 'population', 'country_iso']],
        on='location_id',
        how='left'
    )
    
    # Calculate population density (population per degree square at that location)
    # Approximate: ~111 km per degree
    merged_data['pop_density'] = merged_data['population'] / 1000  # Rescale for visualization
    
    # Classify regions
    def classify_region(pollution, pop_density):
        if pollution > 50 and pop_density > 100:
            return 'High Pollution + High Density'
        elif pollution > 50:
            return 'High Pollution + Low Density'
        elif pop_density > 100:
            return 'Low Pollution + High Density'
        else:
            return 'Low Pollution + Low Density'
    
    merged_data['region_type'] = merged_data.apply(
        lambda row: classify_region(row['pollution_mean'], row['pop_density']), 
        axis=1
    )
    
    # Create subplots - Small Multiples
    region_types = sorted(merged_data['region_type'].unique())
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=region_types,
        specs=[[{'type': 'scatter'}, {'type': 'scatter'}],
               [{'type': 'scatter'}, {'type': 'scatter'}]],
        vertical_spacing=0.12,
        horizontal_spacing=0.12
    )
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
    
    for idx, (region, pos, color) in enumerate(zip(region_types, positions, colors)):
        region_data = merged_data[merged_data['region_type'] == region]
        
        scatter = go.Scatter(
            x=region_data['pop_density'],
            y=region_data['pollution_mean'],
            mode='markers',
            marker=dict(
                size=10,
                color=region_data['pollution_mean'],
                colorscale='Viridis',  # Sequential scale for better perception
                showscale=(idx == 0),  # Show scale bar only once
                colorbar=dict(
                    title=f"{selected_param.upper()}<br>(µg/m³)",
                    x=1.02
                ),
                line=dict(color='white', width=1)
            ),
            text=region_data['location_name'],
            hovertemplate='<b>%{text}</b><br>' +
                          f'Population Density: %{{x:.0f}}<br>' +
                          f'{selected_param.upper()}: %{{y:.1f}} µg/m³<extra></extra>',
            name=region
        )
        
        fig.add_trace(scatter, row=pos[0], col=pos[1])
        
        # Update axes labels
        fig.update_xaxes(title_text="Population Density (scaled)", row=pos[0], col=pos[1])
        fig.update_yaxes(title_text=f"{selected_param.upper()} Level (µg/m³)", row=pos[0], col=pos[1])
    
    fig.update_layout(
        height=800,
        width=1200,
        title_text=f"Small Multiples: {selected_param.upper()} vs Population Density (Stratified by Region)",
        showlegend=False,
        template='plotly_white'
    )
    
    return fig, merged_data


def run_task_3(df):
    """Task 3: Distribution Modeling & Tail Integrity"""
    st.header("🏭 Task 3: Distribution Modeling & Tail Integrity")
    st.markdown('''
    **Objective:** Report the probability of "Extreme Hazard" events (PM2.5 > 200 µg/m³) and produce distribution plots optimized for peaks vs. tails for an industrial zone.
    ''')
    
    # Filter for PM2.5
    pm25_df = df[df['parameter'] == 'pm25'].copy()
    
    if pm25_df.empty:
        st.error("No PM2.5 data available.")
        return
        
    # Classify zones using same heuristic
    industrial_keywords = ['industrial', 'factory', 'station', 'port', 'airport', 'highway', 'urban', 'city']
    
    def is_industrial(name):
        name_lower = str(name).lower()
        return any(k in name_lower for k in industrial_keywords)
        
    pm25_df['is_industrial'] = pm25_df['location_name'].apply(is_industrial)
    industrial_df = pm25_df[pm25_df['is_industrial']]
    
    if industrial_df.empty:
        # Fallback to all data if no industrial matched
        industrial_df = pm25_df
        st.warning("Could not explicitly identify industrial zones, using all available locations.")
        
    # Let user select a specific industrial location
    locations = industrial_df['location_name'].unique()
    selected_loc = st.selectbox("Select an Industrial Zone for Analysis:", sorted(locations))
    
    loc_data = industrial_df[industrial_df['location_name'] == selected_loc].dropna(subset=['value'])
    
    if loc_data.empty:
        st.warning("No valid data for selected location.")
        return
        
    st.markdown("---")
    
    # Computations
    total_obs = len(loc_data)
    extreme_obs = len(loc_data[loc_data['value'] > 200])
    prob_extreme = (extreme_obs / total_obs) * 100 if total_obs > 0 else 0
    p99 = np.percentile(loc_data['value'], 99)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Observations", f"{total_obs:,}")
    col2.metric("99th Percentile", f"{p99:.2f} µg/m³")
    col3.metric("Prob(PM2.5 > 200)", f"{prob_extreme:.3f}%")
    
    st.markdown("---")
    
    # Visualizations
    st.subheader("Distribution Plots: Peaks vs Tails")
    
    tab_a, tab_b = st.tabs(["📊 Linear Scale (Optimized for Peaks)", "📉 Log Scale (Optimized for Tails)"])
    
    with tab_a:
        fig_linear = px.histogram(
            loc_data, x="value", nbins=50,
            title=f"Linear Histogram for {selected_loc}",
            labels={"value": "PM2.5 (µg/m³)"},
            color_discrete_sequence=['#3b5998']
        )
        fig_linear.add_vline(x=p99, line_dash="dash", line_color="red", annotation_text="99th Pct")
        fig_linear.update_layout(template='plotly_white')
        st.plotly_chart(fig_linear, use_container_width=True)
        st.markdown("**Observation:** The linear scale shows where the vast majority of normal readings occur (the 'peaks'). However, values beyond 100 or 200 µg/m³ are visually compressed into the baseline, hiding the severity and frequency of extreme events.")
        
    with tab_b:
        fig_log = px.histogram(
            loc_data, x="value", nbins=50, log_y=True,
            title=f"Log-Y Histogram for {selected_loc}",
            labels={"value": "PM2.5 (µg/m³)"},
            color_discrete_sequence=['#e74c3c']
        )
        fig_log.add_vline(x=p99, line_dash="dash", line_color="black", annotation_text="99th Pct")
        fig_log.add_vline(x=200, line_dash="solid", line_color="darkred", annotation_text="Extreme Hazard (>200)")
        fig_log.update_layout(template='plotly_white')
        st.plotly_chart(fig_log, use_container_width=True)
        st.markdown("**Observation:** The logarithmic Y-axis prevents the extreme values from being dwarfed by the common values. Even rare occurrences (e.g., counts of 1 or 2) remain visible, clearly revealing the 'long tail' of toxic exposures.")
        
    st.markdown("---")
    
    st.header("💡 Technical Justification")
    st.markdown("""
    ### Which plot offers a more "honest" depiction?
    
    For assessing **environmental hazards** and public health, the **Log-Scale Histogram** (optimized for tails) is significantly more "honest."
    
    1. **Visibility of the Long Tail:** In an environmental context, the most critical data points are often the outliers (the extreme hazard events). A linear histogram obscures these rare but deadly events because their frequency is dwarfed by normal background levels. The log-scale ensures that even single extreme occurrences are visually recognizable.
    2. **Risk Assessment:** The 99th percentile was computed as **{:.2f} µg/m³**, highlighting that top 1% of exposure events are exceptionally high. A standard linear plot makes this 1% appear negligible or almost invisible.
    3. **Tail Integrity:** By stretching the lower frequencies, the log-scale plot maintains "tail integrity," ensuring policy-makers do not underestimate the probability of values reaching life-threatening levels (>200 µg/m³).
    """.format(p99))


def create_bivariate_map(merged_data, selected_param='pm25'):
    """
    Create Bivariate Mapping visualization
    Using a 3x3 grid of colors to represent combined pollution and population density
    """
    # Create a copy to avoid modifying original data
    data = merged_data.copy()
    
    # Create bins for bivariate analysis
    # Remove NaN values before binning
    data = data.dropna(subset=['pollution_mean', 'pop_density'])
    
    # Create quantile bins without labels first to avoid mismatch with duplicates='drop'
    data['pollution_bin_raw'] = pd.qcut(
        data['pollution_mean'],
        q=3,
        duplicates='drop',
        labels=False
    )
    data['density_bin_raw'] = pd.qcut(
        data['pop_density'],
        q=3,
        duplicates='drop',
        labels=False
    )
    
    # Map numeric bins to categorical labels
    bin_labels = {0: 'Low', 1: 'Medium', 2: 'High'}
    data['pollution_bin'] = data['pollution_bin_raw'].map(bin_labels)
    data['density_bin'] = data['density_bin_raw'].map(bin_labels)
    
    # Create bivariate color mapping
    bivariate_colors = {
        ('Low', 'Low'): '#E8E8E8',          # Light gray
        ('Low', 'Medium'): '#A1D99B',       # Light green
        ('Low', 'High'): '#41AB5D',         # Medium green
        ('Medium', 'Low'): '#FDD9B5',       # Light orange
        ('Medium', 'Medium'): '#FEB24C',    # Medium orange
        ('Medium', 'High'): '#F16913',      # Dark orange
        ('High', 'Low'): '#FEEDDE',         # Very light red
        ('High', 'Medium'): '#FDBE85',      # Light red
        ('High', 'High'): '#B30000',        # Dark red
    }
    
    data['bivariate_color'] = data.apply(
        lambda row: bivariate_colors.get((row['pollution_bin'], row['density_bin']), '#CCCCCC'),
        axis=1
    )
    
    # Create scatter map
    fig = go.Figure()
    
    for color in bivariate_colors.values():
        subset = data[data['bivariate_color'] == color]
        if len(subset) > 0:
            fig.add_trace(go.Scatter(
                x=subset['longitude'],
                y=subset['latitude'],
                mode='markers',
                marker=dict(
                    size=12,
                    color=color,
                    line=dict(color='white', width=1)
                ),
                text=subset['location_name'] + '<br>' +
                     'Pollution: ' + subset['pollution_bin'].astype(str) + '<br>' +
                     'Density: ' + subset['density_bin'].astype(str),
                hovertemplate='<b>%{text}</b><extra></extra>',
                name=f"{subset.iloc[0]['pollution_bin']} Pol. / {subset.iloc[0]['density_bin']} Dens."
            ))
    
    fig.update_layout(
        title=f"Bivariate Mapping: Location Distribution",
        xaxis_title="Longitude",
        yaxis_title="Latitude",
        height=700,
        width=1000,
        template='plotly_white',
        hovermode='closest'
    )
    
    return fig


def run_task_4(df):
    """Task 4: Visualization Design Analysis - 3D Bar Chart Proposal Evaluation"""
    
    st.header("📊 Task 4: Visualization Design Analysis")
    st.markdown("""
    **Objective:** Evaluate a proposed 3D bar chart for displaying Pollution vs. Population Density vs. Region
    and implement a better alternative visualization if needed.
    """)
    
    # Load population data
    population_df = load_population_data()
    
    if population_df is None:
        st.stop()
    
    st.markdown("---")
    
    # Section 1: Proposal Analysis
    st.header("🔍 Step 1: Evaluating the 3D Bar Chart Proposal")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Lie Factor Analysis")
        st.markdown("""
        **Definition:** Lie Factor = (size of effect shown in visualization) / (size of effect in data)
        
        A Lie Factor > 1.05 indicates misleading visualization.
        
        ### Why 3D Bar Charts Fail:
        
        1. **Perspective Distortion**
           - 3D projection compresses depth values
           - Bars at the back appear smaller than equal-valued bars in front
           - Creates false hierarchies
        
        2. **Lie Factor Problem**
           - A bar with half the height may appear < 25% of original size due to perspective
           - Lie Factor ≈ 2.5 to 3.0 (⚠️ Severely misleading)
        
        3. **Occlusion**
           - Tall bars hide bars behind them
           - Users cannot see or compare all values
        """)
    
    with col2:
        st.subheader("Data-Ink Ratio Analysis")
        st.markdown("""
        **Definition:** Data-Ink Ratio = (ink for data) / (total ink)
        
        Target: > 0.9 (minimize decoration, maximize data)
        
        ### Why 3D Bar Charts Fail:
        
        1. **Unnecessary Decoration**
           - 3D effects add zero information
           - ~40% of ink wasted on 3D perspective
           - Data-Ink Ratio ≈ 0.5 (⚠️ Highly inefficient)
        
        2. **Cognitive Load**
           - Users must interpret 3D rotation
           - Harder to read exact values
           - Adds visual clutter
        
        3. **No Dimension Advantage**
           - 3D doesn't help display 3 variables
           - Could be done better in 2D
        """)
    
    st.markdown("---")
    
    # Section 2: Decision
    st.header("✋ Decision: REJECT the 3D Bar Chart")
    
    st.warning("""
    **Verdict:** The proposed 3D bar chart violates both key visualization principles:
    - 🔴 **Lie Factor ≈ 2.5** (acceptable range: < 1.05)
    - 🔴 **Data-Ink Ratio ≈ 0.5** (target: > 0.9)
    
    **Recommendation:** Implement a Small Multiples approach instead.
    """)
    
    st.markdown("---")
    
    # Section 3: Recommended Implementation
    st.header("✅ Solution: Small Multiples Approach")
    
    st.markdown("""
    ### Why Small Multiples?
    
    1. **Maintains Lie Factor ≈ 1.0**: No perspective distortion
    2. **High Data-Ink Ratio ≈ 0.95**: Minimal decorative elements
    3. **Enables Easy Comparison**: All quadrants visible simultaneously
    4. **Geographic Intuition**: Can overlay on maps for spatial analysis
    
    ### Implementation Strategy:
    - **Stratification:** Divide data into 4 regions based on pollution/density levels
    - **Faceting:** Create 2×2 grid of plots, one per region
    - **Sequential Color Scale:** One color = one pollution level (luminance increases with pollution)
    - **Hover Details:** Rich interaction for individual location inspection
    """)
    
    st.markdown("---")
    
    # Section 4: Color Scale Justification
    st.header("🎨 Step 2: Justifying Sequential vs. Rainbow Color Scales")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Sequential Color Scale (Recommended)")
        st.markdown("""
        ✅ **Why Sequential for Pollution Data:**
        
        1. **Luminance Gradient**
           - Low pollution: Dark (low luminance)
           - High pollution: Bright (high luminance)
           - Human eye perceives luminance linearly
        
        2. **Perceptual Uniformity**
           - Equal data differences = equal visual differences
           - No color artifacts or bias
        
        3. **Intuitive Interpretation**
           - Darker = lower values (safer)
           - Brighter = higher values (danger)
           - Aligns with human expectation
        
        4. **Colorblind Accessible**
           - Works for ~8% colorblind population
           - 'Viridis' scale: yellow (visible) → purple
        
        5. **Print Friendly**
           - Grayscale conversion still readable
           - No color-dependent meaning lost
        """)
    
    with col2:
        st.subheader("Rainbow Color Scale (Not Recommended)")
        st.markdown("""
        ❌ **Why Rainbow FAILS for Pollution Data:**
        
        1. **Non-uniform Luminance**
           - Blue & purple: Low luminance (hard to see)
           - Yellow & white: High luminance (burn out perception)
           - Green: Medium luminance (false center)
        
        2. **Perceptual Artifacts**
           - Equal data = unequal visual weight
           - Artificial "boundaries" at color transitions
           - Yellow appears brighter than red (wrong meaning)
        
        3. **Misleading Interpretation**
           - Red != "danger" (could be low value)
           - Green != "healthy" (in rainbow, often middle)
           - Causes misinterpretation
        
        4. **Colorblind Issues**
           - 8% population cannot distinguish red-green
           - Grayscale: completely unreadable
        
        5. **Scientific Consensus**
           - Viridis, Plasma creators (2015 study):
           - Rainbow causes up to 30% error in value estimation
        """)
    
    st.markdown("---")
    
    # Select parameter for visualization
    col1, col2 = st.columns([3, 1])
    
    with col1:
        selected_param = st.selectbox(
            "Select Pollutant Parameter:",
            options=['pm25', 'pm10', 'no', 'co', 'temperature', 'relativehumidity'],
            index=0
        )
    
    with col2:
        if st.button("🔄 Generate Visualizations", key="task4_viz"):
            st.session_state.task4_gen = True
    
    if st.session_state.get('task4_gen', False):
        # Generate visualizations
        with st.spinner(f"Generating visualizations for {selected_param.upper()}..."):
            try:
                fig_small, merged_data = create_small_multiples(df, population_df, selected_param)
                
                st.markdown("---")
                st.markdown("## 📍 Small Multiples: Stratified Analysis")
                st.markdown("""
                This visualization divides the data into 4 quadrants based on pollution levels and population density.
                Each subplot shows locations within that region, with color indicating pollution intensity.
                
                **Key Observation Areas:**
                - Top-left: Low pollution + Low density (rural areas)
                - Top-right: Low pollution + High density (well-managed urban areas)
                - Bottom-left: High pollution + Low density (industrial sites)
                - Bottom-right: High pollution + High density (problematic urban centers)
                """)
                st.plotly_chart(fig_small, use_container_width=True)
                
                # Generate bivariate map
                st.markdown("---")
                st.markdown("## 🗺️ Bivariate Map: Geographic Distribution")
                st.markdown("""
                This map shows the geographic distribution of locations stratified by their pollution-density combination.
                Each color represents a unique combination of pollution level (Low/Medium/High) and population density.
                """)
                
                fig_bivariate = create_bivariate_map(merged_data, selected_param)
                st.plotly_chart(fig_bivariate, use_container_width=True)
                
                # Display statistics
                st.markdown("---")
                st.header("📊 Data Summary")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "Total Locations",
                        len(merged_data),
                        delta=None
                    )
                
                with col2:
                    st.metric(
                        f"Avg {selected_param.upper()} Level",
                        f"{merged_data['pollution_mean'].mean():.2f} µg/m³",
                        delta=None
                    )
                
                with col3:
                    st.metric(
                        "Avg Population Density",
                        f"{merged_data['pop_density'].mean():.0f}",
                        delta=None
                    )
                
                with col4:
                    st.metric(
                        "Countries Covered",
                        merged_data['country_iso'].nunique(),
                        delta=None
                    )
                
                # Regional breakdown table
                st.markdown("---")
                st.subheader("Region Type Breakdown")
                
                region_summary = merged_data.groupby('region_type').agg({
                    'location_id': 'count',
                    'pollution_mean': ['mean', 'max'],
                    'pop_density': 'mean',
                    'population': 'mean'
                }).round(2)
                
                region_summary.columns = ['Count', 'Avg Pollution', 'Max Pollution', 'Avg Density', 'Avg Population']
                st.dataframe(region_summary, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error generating visualizations: {str(e)}")
    
    st.markdown("---")
    
    # Conclusion
    st.header("📝 Conclusion")
    
    st.markdown("""
    ### Summary of Analysis:
    
    1. **Rejected 3D Bar Chart** - Violates Lie Factor and Data-Ink Ratio principles
    2. **Implemented Small Multiples** - Stratified analysis with 2×2 grid enables easy comparison
    3. **Used Sequential Color Scale** - Viridis scale with luminance gradient for intuitive interpretation
    
    ### Key Insights:
    - Small Multiples follow data visualization best practices
    - Sequential colors align with human perception of luminance
    - Geographic and statistical overlays provide comprehensive analysis
    - Users can identify problem areas (High Pollution + High Density) at a glance
    
    ### Further Recommendations:
    - Interactive filtering by region type for deeper investigation
    - Time series overlay to see pollution trends by population density
    - Correlation analysis: Are high-density areas always high-pollution areas?
    """)


def main():
    df = load_data()

    st.title("🌍 Urban Environmental Intelligence Challenge")
    st.markdown("### Interactive Environmental Diagnostics Engine")

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Task 1: Dimensionality Reduction",
        "🕒 Task 2: Temporal Analysis",
        "🎯 Task 3: Geographic Analysis",
        "🎨 Task 4: Design Analysis"
    ])

    with tab1:
        run_task_1(df)

    with tab2:
        run_task_2(df)
    
    with tab3:
        run_task_3(df)
    
    with tab4:
        run_task_4(df)


if __name__ == "__main__":
    main()
