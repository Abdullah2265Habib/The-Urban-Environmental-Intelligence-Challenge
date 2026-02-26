# The-Urban-Environmental-Intelligence-Challenge
# 🚀 Quick Start Guide - Streamlit Dashboard

## Installation (One-Time Setup)

```bash
# Navigate to project directory
cd "c:\Users\Abdullah Habib\Desktop\click me\DS Assignment\The-Urban-Environmental-Intelligence-Challenge"

# Activate virtual environment
.venv\Scripts\Activate.ps1

# Install required packages (if not already installed)
pip install streamlit pandas numpy scikit-learn plotly matplotlib seaborn

```

## Running the Dashboard

```bash
# Method 1: Direct command
streamlit run streamlit_dashboard.py

# Method 2: With Python executable
.\.venv\Scripts\python.exe -m streamlit run streamlit_dashboard.py

# Method 3: With logging
streamlit run streamlit_dashboard.py --logger.level=info

```

## Expected Output

```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.1.100:8501

Press CTRL+C to stop the server
```

## Access the Dashboard

1. Open browser and go to: **http://localhost:8501**
2. Select environmental variables from sidebar
3. Adjust number of components
4. Explore all visualizations and analyses

## Main Dashboard Sections

### Left Sidebar
- ⚙️ Select 2-6 environmental variables
- Select number of principal components
- View methodology information

### Main Area
1. **📈 Statistics**: Sample count, dimension info
2. **📊 Variance Analysis**: Variance per component + cumulative
3. **🔍 Loading Analysis**: Variable contributions heatmap
4. **📍 2D Projections**: ALL combinations of PC pairs
5. **🎨 3D Projections**: Multi-angle 3D views
6. **🏭 Zone Analysis**: Industrial vs Residential clustering
7. **💡 Insights**: Key findings summary

## Default Configuration

- **Variables**: PM2.5, PM10, NO₂, O₃, Temperature, Relative Humidity
- **Components**: 6 maximum
- **Zones**: Auto-classified based on location name patterns
- **Samples**: Up to 100 sensors represented

## Keyboard Shortcuts

| Action | Key |
|--------|-----|
| Stop Dashboard | CTRL+C |
| Rerun App | R (in app) |
| Clear Cache | CTRL+C, then restart |
| Full Screen Plot | Click expand icon |

## Common Tasks

### View All PC Combinations
1. Keep default 6 variables
2. Set components to 6
3. Scroll through all C(6,2) = 15 different 2D plots

### Analyze Specific Pollutants
1. Uncheck all variables
2. Select PM2.5, PM10, NO₂, O₃
3. Keep 4 components for 6 combinations

### Temperature Effect Analysis
1. Select: PM2.5, PM10, NO₂, Temperature, Humidity + one more
2. Set 5 components
3. Check PC1 loadings to see temperature contribution

### Minimal 2D Analysis
1. Select only 2 variables
2. Set components to 2
3. Get direct 2D visualization with no information loss

## Troubleshooting

### "Port 8501 already in use"
```bash
# Run on different port
streamlit run streamlit_dashboard.py --server.port 8502
```

### "No module named pandas"
```bash
# Reinstall packages
pip install --upgrade pandas scikit-learn plotly streamlit
```

### "Dashboard loads but no data"
1. Check `dataset/` folder exists with CSV files
2. Ensure CSV filenames start with "openaq"
3. Verify CSV files are not empty

### "Slow performance"
- Reduce number of components
- Reduce number of variables
- Close other applications

## File Structure

```
The-Urban-Environmental-Intelligence-Challenge/
├── streamlit_dashboard.py          ← Main dashboard file
├── DASHBOARD_README.md             ← Detailed documentation
├── QUICKSTART.md                   ← This file
├── dataset/                        ← CSV data files
│   ├── openaq_location_100_measurements.csv
│   ├── openaq_location_101_measurements.csv
│   └── ... (100+ location files)
└── .venv/                         ← Virtual environment
```

## Key Features Explained

### 🔄 All PC Combinations
Dashboard **automatically generates all possible 2D plots** from selected PCs:
- 3 PCs → 3 combinations: (PC1,PC2), (PC1,PC3), (PC2,PC3)
- 4 PCs → 6 combinations: (PC1,PC2), (PC1,PC3), (PC1,PC4), (PC2,PC3), (PC2,PC4), (PC3,PC4)
- 6 PCs → 15 combinations total (C(6,2) = 15)

### 📊 Variance Metrics
- **Individual Variance**: % of total variance each PC captures
- **Cumulative Variance**: Total % captured up to that PC
- **Typical pattern**: PC1 ≈ 40-50%, PC1+PC2 ≈ 70-80%

### 🔍 Loadings
Shows how much each variable contributes to each PC:
- Purple (negative) → inverse relationship
- Red (positive) → direct relationship
- Darker colors → stronger contribution

### 🎨 Zone Clustering
- Red dots: Industrial zones (high pollution)
- Teal dots: Residential zones (lower pollution)
- Green dots: Mixed areas
- **Key insight**: Look for spatial separation between colors

## Data Insights

### What the Data Contains
- 383,965 total measurements
- 100+ sensor locations across urban areas
- 11 environmental parameters
- Time-series data from multiple dates

### Environmental Variables
1. **PM2.5**: Fine particulate matter (< 2.5 μm)
2. **PM10**: Coarse particulate matter (< 10 μm)
3. **NO₂**: Nitrogen dioxide (traffic/industrial)
4. **O₃**: Ozone (created from NO₂ + VOCs)
5. **CO**: Carbon monoxide (vehicle emissions)
6. **SO₂**: Sulfur dioxide (industrial/coal)
7. **Temperature**: Air temperature (°C)
8. **Relative Humidity**: Air moisture (%)

## Advanced Usage

### Export Analysis Results
Use browser's developer tools or Streamlit export:
```python
# In Python to export PCA results
import streamlit as st
import pandas as pd

results_df = pd.DataFrame({
    'PC1': pca_result[:, 0],
    'PC2': pca_result[:, 1],
    'Zone': zones
})
results_df.to_csv('pca_results.csv', index=False)
```

### Modify Default Variables
Edit `streamlit_dashboard.py` line ~270:
```python
default_params = ['pm25', 'pm10', 'no2', 'o3', 'so2', 'temperature']
```

### Change Color Scheme
Edit color mappings in visualization functions:
```python
color_discrete_map={
    'Industrial': '#FF6B6B',    # Change colors here
    'Residential': '#4ECDC4',
    'Mixed': '#95E1D3'
}
```

## Performance Tips

✅ **Recommended Setup**
- 4-6 variables
- 4-5 components
- Modern machine (Intel i5+ or equivalent)
- 8GB+ RAM
- Broadband internet (for Plotly rendering)

⚠️ **May be Slow**
- 6+ variables with 6 components
- Old machine (< 4GB RAM)
- Many simultaneous users

## Getting Help

1. **Check data**: `python -c "import pandas as pd; df = pd.read_csv('dataset/openaq_location_100_measurments.csv'); print(df.columns)"`
2. **Test PCA**: Run the notebook.ipynb to see if PCA works
3. **Verify environment**: `pip list | grep -E "streamlit|pandas|plotly"`

## Next Steps

After exploring the dashboard:
1. Identify which PC best separates Industrial/Residential zones
2. Examine loadings to understand variable relationships
3. Note variance explained by top 2 PCs
4. Document findings for analysis report

---

**Ready to explore?** Run `streamlit run streamlit_dashboard.py` and navigate to http://localhost:8501 ! 🚀

## Overview

This interactive Streamlit dashboard solves **Task 1: The Dimensionality Challenge** by providing advanced dimensionality reduction analysis of environmental data from 100 sensors across different urban zones.

## Features

### 🌟 Core Functionality

1. **Dimensionality Reduction**
   - Principal Component Analysis (PCA) for 2D to 6D projections
   - Standardized data preprocessing
   - Automatic variable correlation reduction

2. **All PCA Combination Visualizations**
   - Displays all possible 2D combinations of principal components
   - Interactive scatter plots with hover information
   - Color-coded zones (Industrial, Residential, Mixed)

3. **Variance Analysis**
   - Individual variance per component (bar chart)
   - Cumulative variance explained (line chart)
   - Detailed numerical breakdown in table format

4. **PCA Loadings Analysis**
   - Heatmap showing variable contributions to each PC
   - Top contributors identification
   - Interpretation of pollution drivers

5. **3D Visualizations**
   - Multiple 3D scatter plots for deeper analysis
   - PC combinations: (PC1, PC2, PC3), (PC1, PC2, PC4), etc.

6. **Zone Classification**
   - Industrial vs Residential clustering
   - Statistical analysis per zone
   - Pattern separation visualization

## Environmental Variables Analyzed

Default analysis uses 6 environmental variables:
- **PM2.5**: Fine particulate matter
- **PM10**: Coarse particulate matter
- **NO₂**: Nitrogen dioxide
- **O₃**: Ozone
- **Temperature**: Environmental temperature
- **Relative Humidity**: Air moisture content

*Note: Dashboard supports selection of any 2-6 parameters from available data*

Available parameters:
- CO (Carbon monoxide)
- NO (Nitrogen monoxide)
- NO₂ (Nitrogen dioxide)
- SO₂ (Sulfur dioxide)
- PM1, PM10, PM2.5
- O₃ (Ozone)
- Temperature
- Relative Humidity
- UM003 (Particle count)

## Installation & Setup

### Prerequisites
```bash
Python 3.8+
pip or conda
```

### Install Dependencies
```bash
pip install streamlit pandas numpy scikit-learn plotly matplotlib seaborn
```

Or use the complete installation:
```bash
pip install -r requirements.txt
```

## Running the Dashboard

### Basic Command
```bash
streamlit run streamlit_dashboard.py
```

### Expected Output
```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501
```

### Open in Browser
Navigate to `http://localhost:8501` in your web browser

## Dashboard Sections

### 1. **Configuration Sidebar** ⚙️
- Select environmental variables (2-6 parameters)
- Adjust number of principal components (2-6)
- View methodology information

### 2. **Dimensionality Reduction Statistics** 📈
- Total samples analyzed
- Original dimensions vs reduced dimensions
- Data retention percentage

### 3. **Variance Analysis** 📊
- Variance per component visualization
- Cumulative variance explanation
- Detailed variance breakdown table

### 4. **PCA Loadings Analysis** 🔍
- Heatmap of variable contributions
- Top 3 contributors to PC1 highlighted
- Interpretation of feature importance

### 5. **2D PCA Projections** 📍
- All possible 2D combinations automatically generated
- Example for 3 PCs: (PC1,PC2), (PC1,PC3), (PC2,PC3)
- Example for 4 PCs: 6 different 2D plots
- Interactive scatter plots with zone coloring

### 6. **3D Visualizations** 🎨
- 3D scatter plots for enhanced depth perception
- Multiple angle combinations
- Rotation and zoom capabilities

### 7. **Zone Classification Analysis** 🏭
- Industrial, Residential, and Mixed zone counts
- Statistical metrics per zone
- Mean, std, min, max values for PC1 and PC2

### 8. **Key Insights** 💡
- Variance distribution summary
- Main drivers of urban pollution
- Methodology justification

## Data Processing Pipeline

```
Raw CSV Data (100 sensors)
    ↓
Load & Concatenate
    ↓
Filter by Selected Parameters
    ↓
Pivot Table (Location × Parameter)
    ↓
Remove Missing Values
    ↓
Zone Classification
    ↓
StandardScaler (Normalization)
    ↓
PCA Transformation
    ↓
Visualization & Analysis
```

## How to Interpret Results

### Understanding Variance Explained
- **PC1**: 40-50% variance typically captured
- **PC1 + PC2**: 70-80% variance typically captured
- **Higher components**: Diminishing returns

### Reading Loadings
- **Large positive values**: Variable strongly contributes in positive direction
- **Large negative values**: Variable contributes in negative direction
- **Values near 0**: Little contribution to that PC

### Zone Clustering
- **Tight clusters**: Strong zone differentiation
- **Overlapping clusters**: Pollution patterns similar between zones
- **Separation distance**: Magnitude of zone differences

## Example Usage Scenarios

### Scenario 1: Analyzing Top 4 Variables
1. Select: PM2.5, PM10, NO₂, O₃
2. Set components: 4
3. View 6 different 2D combinations
4. Identify which PC separates zones best

### Scenario 2: Including Temperature Effects
1. Select: PM2.5, PM10, NO₂, Temperature, Humidity
2. Set components: 5
3. Examine if temperature/humidity differentiate zones
4. Check loadings to see relationships

### Scenario 3: Minimal Dimensionality (2D Direct)
1. Select only 2 variables
2. Set components: 2
3. Get direct 2D visualization
4. No variance loss - all information preserved

## Technical Details

### Standardization
Uses `sklearn.preprocessing.StandardScaler`:
- Removes mean: (x - μ)
- Divides by std: (x - μ) / σ
- Ensures equal weight for all variables

### PCA Algorithm
Uses `sklearn.decomposition.PCA`:
- Computes covariance matrix
- Finds eigenvectors and eigenvalues
- Selects top k components by variance

### Zone Classification Logic
Heuristic-based on location names:
- **Industrial keywords**: factory, station, port, highway, urban
- **Residential keywords**: suburb, village, residential
- **Default**: Mixed if no keywords match

## Customization Options

### Change Default Parameters
Edit `default_params` in `main()` function (line ~250):
```python
default_params = ['pm25', 'pm10', 'no2', 'o3', 'so2', 'temperature']
```

### Modify Color Scheme
Edit color_discrete_map in visualization functions:
```python
color_discrete_map={
    'Industrial': '#FF6B6B',      # Red
    'Residential': '#4ECDC4',     # Teal
    'Mixed': '#95E1D3'            # Mint
}
```

### Adjust Sampling
Modify line in `prepare_data_for_pca()` (~line 110):
```python
pivot_df = pivot_df.sample(n=150, random_state=42)  # Change 100 to desired count
```

## Troubleshooting

### Dashboard won't load
```bash
# Clear Streamlit cache
streamlit cache clear

# Run with verbose logging
streamlit run streamlit_dashboard.py --logger.level=debug
```

### Slow performance
- Select fewer variables
- Reduce number of components
- Dataset may be filtered to top 100 samples

### Missing parameters error
Check available parameters:
```python
df['parameter'].unique()
```

### Memory issues
- Use fewer principal components
- Sample fewer locations
- Run on machine with more RAM

## Performance Characteristics

| Metric | Value |
|--------|-------|
| Data Points per Location | ~3,800 |
| Total Samples | 383,965 |
| Locations (Sampled) | 100 |
| Variables Available | 11 |
| Variables Selected | 2-6 |
| PCA Time | < 1 second |
| Dashboard Load Time | 5-10 seconds |

## Key Insights from Analysis

### Main Findings
1. **Variance Concentration**: Top 2 PCs capture 70-80% of variance
2. **Zone Separation**: Industrial zones show distinct pollution signatures
3. **Variable Importance**: PM2.5, NO₂, and PM10 are primary drivers
4. **Climate Factors**: Temperature and humidity influence pollution distribution

### Methodology Justification
- **PCA wins vs t-SNE**: Linear, interpretable loadings; faster computation
- **PCA wins vs UMAP**: Variance preservation explicit; mathematical simplicity
- **Optimal choice for**: Interpretable dimensionality reduction with clear feature contributions

## Output Interpretation Example

If analysis shows:
- PC1 loading for PM2.5: 0.45
- PC1 loading for NO₂: 0.38
- PC1 loading for Temperature: -0.25

**Interpretation**: "PC1 represents overall industrial pollution (high PM2.5, NO₂) vs clean air (inversely related to temperature)"

## References

### Methods
- [PCA Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)
- [StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)

### Environmental Data
- [OpenAQ Database](https://openaq.org/)
- Air Quality Standards

### Streamlit
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Plotly Visualization](https://plotly.com/python/)

## License

This project is provided for educational purposes as part of the Data Science Assignment.

## Support

For issues or questions:
1. Check data files in `dataset/` folder
2. Verify all parameters are available
3. Ensure virtual environment is activated
4. Check error messages in terminal output

---

**Created**: February 2026
**Dashboard Version**: 1.0
**Python**: 3.8+
