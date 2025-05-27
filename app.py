import streamlit as st
import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
from pyproj import Transformer
import branca.colormap as cm
from sklearn.cluster import DBSCAN

# Set page config
st.set_page_config(
    page_title="Passenger Density Analysis",
    page_icon="🦖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for sidebar styling and statistics boxes
st.markdown("""
<style>
    .css-1d391kg, .css-1lcbmhc, .css-1cypcdb, .css-1lcbmhc, section[data-testid="stSidebar"] {
        background-color: #E8F5E8 !important;
    }
    
    .sidebar .sidebar-content {
        background-color: #E8F5E8 !important;
    }
    
    .legend {
        background-color: white;
        border: 2px solid grey;
        border-radius: 5px;
        padding: 10px;
        font-size: 12px;
        line-height: 18px;
    }
    
    .stat-box {
        background-color: #3b8777;
        padding: 12px 8px;
        border-radius: 8px;
        text-align: center;
        margin: 3px 0px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        height: 80px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    
    .stat-value {
        color: white;
        font-size: 20px;
        font-weight: bold;
        margin: 0;
        line-height: 1.2;
    }
    
    .stat-label {
        color: white;
        font-size: 11px;
        margin: 4px 0 0 0;
        line-height: 1.1;
    }
    
    .compact-title {
        font-size: 24px;
        margin-bottom: 10px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Compact title
st.markdown('<h1 class="compact-title" style="font-size: 28px;">Passenger Density - Cagayan de Oro</h1>', unsafe_allow_html=True)

# Load and cache data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('cleaned_data.csv')
        transformer = Transformer.from_crs("EPSG:3125", "EPSG:4326")
        lat, lon = transformer.transform(df['longitude'], df['latitude'])
        df['lat_wgs84'], df['lon_wgs84'] = lat, lon
        df = df.dropna(subset=['lat_wgs84', 'lon_wgs84', 'total_passenger'])

        if 'time_slot' not in df.columns:
            st.error("Column 'time_slot' not found in the dataset")
            return None

        # Check for required columns
        required_columns = ['jeep_count', 'alley_capacity']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            st.error(f"Missing required columns: {missing_columns}")
            return None

        return df
    except FileNotFoundError:
        st.error("File 'cleaned_data.csv' not found. Please upload the file to the same directory as app.py")
        return None
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

df = load_data()

if df is not None:
    st.sidebar.header("Controls")

    desired_time_order = [
        '8:00:00 am to 10:00:00 am',
        '10:00:00 am to 12:00:00 pm',
        '12:00:00 pm to 3:00:00 pm',
        '3:00:00 pm to 5:00:00 pm'
    ]

    available_time_slots = df['time_slot'].unique()
    time_slots = [slot for slot in desired_time_order if slot in available_time_slots]
    additional_slots = [slot for slot in available_time_slots if slot not in desired_time_order]
    time_slots.extend(sorted(additional_slots))

    # TIME SLOT SELECTION
    if len(time_slots) > 1:
        selected_time_slot_idx = st.sidebar.slider(
            "Select Time Slot:",
            min_value=0,
            max_value=len(time_slots) - 1,
            value=0,
            help="Slide to navigate through different time periods"
        )
        selected_time_slot = time_slots[selected_time_slot_idx]
        st.sidebar.write(f"**Current Time Slot:**")
        st.sidebar.write(f"📅 {selected_time_slot}")
    else:
        selected_time_slot = time_slots[0]
        st.sidebar.write(f"**Time Slot:**")
        st.sidebar.write(f"📅 {selected_time_slot}")

    # MAP STYLE
    map_style = st.sidebar.selectbox(
        "Map Style:",
        options=['OpenStreetMap', 'CartoDB dark_matter', 'CartoDB positron', 'Stamen Terrain'],
        index=0
    )

    st.sidebar.subheader("Heatmap Settings")
    
    # EXISTING CONTROLS
    radius = st.sidebar.slider("Heatmap Radius", min_value=10, max_value=50, value=40)
    blur = st.sidebar.slider("Heatmap Blur", min_value=5, max_value=30, value=13)
    max_zoom = st.sidebar.slider("Max Zoom Level", min_value=10, max_value=18, value=13)
    
    # NEW CONTROLS - INTERPRETABILITY FEATURES
    st.sidebar.subheader("Display Options")
    
    show_markers = st.sidebar.checkbox("Show Location Markers", value=False,
                                      help="Show individual location markers with details")
    
    show_heatmap = st.sidebar.checkbox("Show Heatmap", value=True,
                                      help="Show the kernel density estimation heatmap")
    
    heatmap_type = st.sidebar.radio(
        "Heatmap Type:",
        ["Passenger Density"],
        help="Choose what to visualize in the heatmap"
    )
    
    # WEIGHTED DATA CONTROL
    st.sidebar.subheader("Data Filtering")
    
    # Get passenger count range for the selected time slot
    filtered_df_temp = df[df['time_slot'] == selected_time_slot].copy()
    min_passengers = int(filtered_df_temp['total_passenger'].min())
    max_passengers = int(filtered_df_temp['total_passenger'].max())
    
    passenger_range = st.sidebar.slider(
        "Passenger Count Range",
        min_value=min_passengers,
        max_value=max_passengers,
        value=(min_passengers, max_passengers),
        help="Filter locations by passenger count to focus on specific density levels"
    )

    # Apply filters
    filtered_df = df[
        (df['time_slot'] == selected_time_slot) & 
        (df['total_passenger'] >= passenger_range[0]) & 
        (df['total_passenger'] <= passenger_range[1])
    ].copy()

    # Show filtering info
    if passenger_range != (min_passengers, max_passengers):
        st.info(f"Showing locations with {passenger_range[0]}-{passenger_range[1]} passengers")

    # IMPROVED LOAD FACTOR AND CALCULATION METHOD
    st.sidebar.subheader("Capacity Analysis Settings")
    
    # Load factor selection - keeping it low to avoid oversupply
    load_factor = st.sidebar.select_slider(
        "Jeepney Load Factor:",
        options=[0.2, 0.3, 0.4, 0.5, 0.6],
        value=0.3,
        format_func=lambda x: f"{int(x*100)}% occupied",
        help="Effective passenger capacity per jeepney (lower = more realistic for actual boarding)"
    )
    
    # Baseline adjustment factor
    baseline_adjustment = st.sidebar.slider(
        "Baseline Adjustment:",
        min_value=0.5,
        max_value=2.0,
        value=1.0,
        step=0.1,
        help="Adjust the baseline for imbalance calculation (1.0 = neutral, <1.0 = more lenient, >1.0 = stricter)"
    )
    
    # Calculation method selection
    calculation_method = st.sidebar.selectbox(
        "Calculation Method:",
        options=["Improved Ratio", "Simple Ratio", "Demand-Supply Gap"],
        index=0,
        help="Choose how to calculate service balance"
    )

    # IMPROVED CALCULATION FUNCTION
    def calculate_imbalance_improved(df, load_factor=0.8, 
                                   time_slot_duration=120, 
                                   dwell_time_per_jeep=5, 
                                   passenger_capacity=22,
                                   baseline_adjustment=1.0,
                                   method="Improved Ratio"):
        df = df.copy()
        
        # Handle zero/missing values
        df['jeep_count'] = df['jeep_count'].replace(0, np.nan).fillna(0)
        df['alley_capacity'] = df['alley_capacity'].replace(0, 1)  # Minimum capacity of 1
        
        # Calculate maximum boarding capacity
        df['max_boarding_capacity'] = df['alley_capacity'] * (time_slot_duration / dwell_time_per_jeep)
        
        # Calculate effective jeep arrivals (limited by boarding capacity)
        df['effective_jeep_arrivals'] = np.minimum(df['jeep_count'], df['max_boarding_capacity'])
        
        # Calculate passenger supply with load factor
        df['passenger_supply'] = df['effective_jeep_arrivals'] * passenger_capacity * load_factor
        
        # Apply baseline adjustment
        df['adjusted_supply'] = df['passenger_supply'] * baseline_adjustment
        
        if method == "Improved Ratio":
            # Improved ratio that's more balanced
            # Uses a different denominator to avoid extreme values
            df['imbalance_ratio'] = (df['total_passenger'] - df['adjusted_supply']) / (
                df['total_passenger'] + df['adjusted_supply'] + 1  # Add 1 to avoid division by zero
            )
            
        elif method == "Simple Ratio":
            # Simple demand/supply ratio
            df['imbalance_ratio'] = np.where(
                df['adjusted_supply'] > 0,
                (df['total_passenger'] / df['adjusted_supply']) - 1,
                1  # If no supply, demand is unmet
            )
            
        elif method == "Demand-Supply Gap":
            # Normalized gap method
            max_demand = df['total_passenger'].max()
            df['imbalance_ratio'] = (df['total_passenger'] - df['adjusted_supply']) / max_demand
        
        # Clean and bound the ratio
        df['imbalance_ratio'] = df['imbalance_ratio'].replace([np.inf, -np.inf], 0).fillna(0)
        df['imbalance_ratio'] = np.clip(df['imbalance_ratio'], -1, 1)
        
        # Much more lenient classification thresholds
        def classify_imbalance_improved(ratio, method_type):
            if method_type == "Improved Ratio":
                if ratio < -0.99:
                    return "Significantly Overserved"
                elif -0.99 <= ratio < -0.1:
                    return "Slightly Overserved"
                elif -0.1 <= ratio <= 0.1:
                    return "Well Balanced"
                elif 0.1 < ratio <= 0.3:
                    return "Slightly Underserved"
                else:
                    return "Significantly Underserved"
            else:  # For other methods
                if ratio < -0.99:
                    return "Significantly Overserved"
                elif -0.99 <= ratio < -0.1:
                    return "Slightly Overserved"
                elif -0.1 <= ratio <= 0.1:
                    return "Well Balanced"
                elif 0.1 < ratio <= 0.3:
                    return "Slightly Underserved"
                else:
                    return "Significantly Underserved"
        
        df['imbalance_status'] = df['imbalance_ratio'].apply(
            lambda x: classify_imbalance_improved(x, method)
        )
        
        return df

    # Apply improved imbalance calculation
    filtered_df = calculate_imbalance_improved(
        filtered_df, 
        load_factor=load_factor,
        baseline_adjustment=baseline_adjustment,
        method=calculation_method
    )

    # MAP AND STATISTICS SECTION (side by side layout)
    map_col, stats_col = st.columns([2, 1], gap="medium")
    
    with map_col:
        st.markdown(f"<h3 style='font-size: 18px;'>🕑 Time Slot: {selected_time_slot}</h3>", unsafe_allow_html=True)
        
        center_lat = filtered_df['lat_wgs84'].mean()
        center_lon = filtered_df['lon_wgs84'].mean()

        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=15,
            tiles=map_style
        )

        # Default color gradient
        default_gradient = {0.2: 'blue', 0.35: 'teal', 0.5: '#00AA80', 0.65: 'yellow', 0.8: '#CC5500'}

        heatmap_data = []
        for _, row in filtered_df.iterrows():
            try:
                lat = float(row['lat_wgs84'])
                lon = float(row['lon_wgs84'])
                weight = float(row['total_passenger'])
                if not (np.isnan(lat) or np.isnan(lon) or np.isnan(weight)):
                    heatmap_data.append([lat, lon, weight])
            except (ValueError, TypeError):  
                continue

        # Add color scale bar if we have data
        if not filtered_df.empty:
            if heatmap_type == "Passenger Density":
                min_val = filtered_df['total_passenger'].min()
                max_val = filtered_df['total_passenger'].max()
                colormap = cm.LinearColormap(
                    colors=['blue', 'teal', '#00AA80', 'yellow', '#CC5500'],
                    vmin=min_val,
                    vmax=max_val,
                    caption='Passenger Density'
                )
            else:  # Service Balance
                min_val = filtered_df['imbalance_ratio'].min()
                max_val = filtered_df['imbalance_ratio'].max()
                colormap = cm.LinearColormap(
                    colors=['blue', 'green', 'red'],  # Blue (overserved) -> Green (balanced) -> Red (underserved)
                    vmin=min_val,
                    vmax=max_val,
                    caption='Service Balance'
                )
            colormap.add_to(m)

        # Add heatmap if enabled
        if heatmap_data and show_heatmap:
            if heatmap_type == "Passenger Density":
                # Original passenger density heatmap
                HeatMap(
                    data=heatmap_data,
                    radius=radius,
                    blur=blur,
                    max_zoom=max_zoom,
                    gradient=default_gradient
                ).add_to(m)

        # Add markers if enabled
        if show_markers:
            marker_group = folium.FeatureGroup(name="Passenger Locations")

            # Create a color scale based on the provided image gradient (Light Green/Yellow -> Medium Teal -> Dark Teal/Blue)
            min_ratio = filtered_df['imbalance_ratio'].min()
            max_ratio = filtered_df['imbalance_ratio'].max()
            
            # Create a color scale based on the provided image gradient (Light Green/Yellow -> Medium Teal -> Dark Teal/Blue)
            def get_color(ratio, min_ratio, max_ratio):
                # Handle cases where min and max are the same to avoid division by zero
                if max_ratio == min_ratio:
                    # Return a default color, e.g., the middle color of the gradient
                    return '#4db6ac' # Medium Teal

                # Normalize ratio to 0-1 range
                normalized = (ratio - min_ratio) / (max_ratio - min_ratio)

                # Define the key colors from the image gradient (approximate hex codes)
                color_stops = {
                    0.0: '#c8e6c9',    # Light green/yellow (Overserved)
                    0.5: '#4db6ac',    # Medium green/teal (Balanced)
                    1.0: '#004d40'     # Dark teal/blue (Underserved)
                }

                # Interpolate colors
                if normalized <= 0.5:
                    # Interpolate between start and middle color
                    start_color = np.array([int(color_stops[0.0][i:i+2], 16) for i in (1, 3, 5)])
                    mid_color = np.array([int(color_stops[0.5][i:i+2], 16) for i in (1, 3, 5)])
                    interpolated_color = start_color + (mid_color - start_color) * (normalized / 0.5)
                else:
                    # Interpolate between middle and end color
                    mid_color = np.array([int(color_stops[0.5][i:i+2], 16) for i in (1, 3, 5)])
                    end_color = np.array([int(color_stops[1.0][i:i+2], 16) for i in (1, 3, 5)])
                    interpolated_color = mid_color + (end_color - mid_color) * ((normalized - 0.5) / 0.5)

                # Ensure values are within 0-255 and convert to hex
                r, g, b = np.clip(interpolated_color, 0, 255).astype(int)
                return f'#{r:02x}{g:02x}{b:02x}'

            # Add color scale legend
            legend_html = f"""
            <div style="position: fixed; bottom: 50px; left: 50px; z-index: 1000; background-color: white; 
            padding: 15px; border: 2px solid grey; border-radius: 5px; font-size: 12px; box-shadow: 0 0 10px rgba(0,0,0,0.2);">
                <p style="margin: 0 0 10px 0; font-size: 14px; font-weight: bold;">Service Balance Scale (Markers)</p>
                <div style="display: flex; flex-direction: column; gap: 5px;">
                    <div style="display: flex; align-items: center;">
                        <div style="width: 150px; height: 15px; background: linear-gradient(to right, 
                            #c8e6c9, #4db6ac, #004d40); margin-right: 10px;"></div>
                    </div>
                    <div style="display: flex; justify-content: space-between; width: 150px;">
                        <span style="color: #c8e6c9;">Overserved</span>
                        <span style="color: #4db6ac;">Balanced</span>
                        <span style="color: #004d40;">Underserved</span>
                    </div>
                    <div style="margin-top: 5px; font-size: 11px; color: #666;">
                        <p style="margin: 0;">• Color intensity indicates degree of imbalance</p>
                        <p style="margin: 0;">• Light colors = Overserved (more supply than demand)</p>
                        <p style="margin: 0;">• Medium colors = Balanced (near zero imbalance)</p>
                        <p style="margin: 0;">• Dark colors = Underserved (more demand than supply)</p>
                        <p style="margin: 0;">• Marker size indicates number of nearby points (hotspot)</p>
                    </div>
                </div>
            </div>
            """
            m.get_root().html.add_child(folium.Element(legend_html))

            # Sort the DataFrame by imbalance_ratio in ascending order for drawing order
            sorted_df = filtered_df.sort_values(by='imbalance_ratio', ascending=True).copy()

            for _, row in sorted_df.iterrows():
                try:
                    lat = float(row['lat_wgs84'])
                    lon = float(row['lon_wgs84'])
                    passengers = int(row['total_passenger'])
                    
                    # Get the calculated values
                    imbalance_ratio = row['imbalance_ratio']
                    passenger_supply = row.get('passenger_supply', 0) if not np.isnan(row.get('passenger_supply', 0)) else 0
                    adjusted_supply = row.get('adjusted_supply', 0) if not np.isnan(row.get('adjusted_supply', 0)) else 0
                    jeep_count = row['jeep_count'] if not np.isnan(row['jeep_count']) else 0
                    effective_jeep_arrivals = row['effective_jeep_arrivals'] if not np.isnan(row['effective_jeep_arrivals']) else 0
                    alley_capacity = row['alley_capacity'] if not np.isnan(row['alley_capacity']) else 0
                    max_boarding_capacity = row.get('max_boarding_capacity', 0) if not np.isnan(row.get('max_boarding_capacity', 0)) else 0
                    
                    # Get color based on imbalance ratio
                    status_color = get_color(imbalance_ratio, min_ratio, max_ratio)
                    
                    if not (np.isnan(lat) or np.isnan(lon)):
                        popup_html = f"""
                        <div style="width: 280px; font-family: Arial, sans-serif;">
                            <h4 style="margin: 5px 0; color: #2c3e50; font-size: 14px;">
                                <b>{str(row.get('name', 'Location'))}</b>
                            </h4>
                            <hr style="margin: 8px 0; border: 1px solid #bdc3c7;">
                            <p style="margin: 4px 0; font-size: 12px;">
                                <b>Passengers:</b> <span style="color: #e74c3c; font-weight: bold;">{passengers}</span>
                            </p>
                            <p style="margin: 4px 0; font-size: 12px;">
                                <b>Jeep Arrivals:</b> <span style="color: #8e44ad; font-weight: bold;">{int(jeep_count)}</span>
                            </p>
                            <p style="margin: 4px 0; font-size: 12px;">
                                <b>Alley Capacity:</b> <span style="color: #3498db; font-weight: bold;">{int(alley_capacity)}</span>
                            </p>
                            <p style="margin: 4px 0; font-size: 12px;">
                                <b>Max Boarding Cap:</b> <span style="color: #f39c12; font-weight: bold;">{max_boarding_capacity:.1f}</span>
                            </p>
                            <p style="margin: 4px 0; font-size: 12px;">
                                <b>Effective Arrivals:</b> <span style="color: #16a085; font-weight: bold;">{effective_jeep_arrivals:.1f}</span>
                            </p>
                            <p style="margin: 4px 0; font-size: 12px;">
                                <b>Passenger Supply:</b> <span style="color: #9b59b6; font-weight: bold;">{passenger_supply:.1f}</span>
                            </p>
                            <p style="margin: 4px 0; font-size: 12px;">
                                <b>Adjusted Supply:</b> <span style="color: #e67e22; font-weight: bold;">{adjusted_supply:.1f}</span>
                            </p>
                            <hr style="margin: 6px 0; border: 1px solid #ecf0f1;">
                            <p style="margin: 4px 0; font-size: 12px;">
                                <b>Imbalance Ratio:</b> <span style="color: #2c3e50; font-weight: bold;">{imbalance_ratio:.3f}</span>
                            </p>
                            <p style="margin: 4px 0; font-size: 12px;">
                                <b>Service Status:</b> <span style="color: {status_color}; font-weight: bold;">{row['imbalance_status']}</span>
                            </p>
                            <hr style="margin: 6px 0; border: 1px solid #ecf0f1;">
                            <p style="margin: 4px 0; font-size: 12px;">
                                <b>Settings:</b><br>
                                <span style="font-size: 11px;">Load Factor: {int(load_factor*100)}%, Method: {calculation_method}</span>
                            </p>
                        </div>
                        """
                        marker = folium.CircleMarker(
                            location=[lat, lon],
                            radius=22, # Further increased size
                            color=status_color,
                            fill=True,
                            fillColor=status_color,
                            fill_opacity=0.9, # Increased opacity for a more 'glowing' effect
                            weight=0, # Removed outline
                            popup=folium.Popup(popup_html, max_width=260)
                        )
                        marker.add_to(marker_group)
                except (ValueError, TypeError):
                    continue

            marker_group.add_to(m)

        map_data = st_folium(m, width=800, height=550)
    
    with stats_col:
        st.subheader("📅 Statistics")
        
        # Calculate statistics
        total_passengers = filtered_df['total_passenger'].sum()
        avg_passengers = filtered_df['total_passenger'].mean()
        max_passengers_filtered = filtered_df['total_passenger'].max()
        num_locations = len(filtered_df)

        # Create 2x2 grid for statistics
        stat_col1, stat_col2 = st.columns(2, gap="small")
        
        with stat_col1:
            st.markdown(f"""
            <div class="stat-box">
                <p class="stat-value">{total_passengers:,}</p>
                <p class="stat-label">Total Passengers</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="stat-box">
                <p class="stat-value">{max_passengers_filtered:,}</p>
                <p class="stat-label">Maximum at One Location</p>
            </div>
            """, unsafe_allow_html=True)
        
        with stat_col2:
            st.markdown(f"""
            <div class="stat-box">
                <p class="stat-value">{avg_passengers:.1f}</p>
                <p class="stat-label">Average per Location</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="stat-box">
                <p class="stat-value">{num_locations:,}</p>
                <p class="stat-label">Number of Locations</p>
            </div>
            """, unsafe_allow_html=True)

        # Service status distribution
        st.subheader("🎯 Service Balance")
        status_counts = filtered_df['imbalance_status'].value_counts()
        for status, count in status_counts.items():
            percentage = (count / len(filtered_df)) * 100
            st.write(f"**{status}**: {count} ({percentage:.1f}%)")

        # Show filtered entries in a scrollable table
        st.subheader("📈 All Locations")
        if not filtered_df.empty:
            # Prepare table
            table_df = filtered_df[[
                'name', 'imbalance_status', 'total_passenger', 'jeep_count', 'alley_capacity',
                'effective_jeep_arrivals', 'passenger_supply', 'adjusted_supply', 'imbalance_ratio'
            ]].copy()
            table_df = table_df.rename(columns={
                'name': 'Location',
                'imbalance_status': 'Service Status',
                'total_passenger': 'Passengers',
                'jeep_count': 'Jeep Arrivals',
                'alley_capacity': 'Alley Cap',
                'effective_jeep_arrivals': 'Effective Arrivals',
                'passenger_supply': 'Base Supply',
                'adjusted_supply': 'Adjusted Supply',
                'imbalance_ratio': 'Imbalance Ratio'
            })
            
            # Round numerical columns
            for col in ['Effective Arrivals', 'Base Supply', 'Adjusted Supply']:
                if col in table_df.columns:
                    table_df[col] = table_df[col].round(1)
            table_df['Imbalance Ratio'] = table_df['Imbalance Ratio'].round(3)

            st.dataframe(
                table_df,
                use_container_width=True,
                hide_index=True,
                height=300
            )
        else:
            st.write("No data available for the selected filters.")

    # Updated explanation
    with st.expander("Understanding the Improved Imbalance Calculation"):
        st.markdown(f"""
        **Current Settings:**
        - **Load Factor**: {int(load_factor*100)}% (jeepneys are {int(load_factor*100)}% occupied on average)
        - **Baseline Adjustment**: {baseline_adjustment}x (supply multiplier)
        - **Calculation Method**: {calculation_method}
        
        **How it works:**
        1. **Max Boarding Capacity** = Alley Capacity × (120 min ÷ 5 min dwell time)
        2. **Effective Jeep Arrivals** = min(Actual Arrivals, Max Boarding Capacity)  
        3. **Base Supply** = Effective Arrivals × 22 passengers × {load_factor} load factor
        4. **Adjusted Supply** = Base Supply × {baseline_adjustment} baseline adjustment
        
        **Calculation Methods:**
        - **Improved Ratio**: (Demand - Supply) ÷ (Demand + Supply + 1) - More balanced
        - **Simple Ratio**: (Demand ÷ Supply) - 1 - Traditional approach
        - **Demand-Supply Gap**: (Demand - Supply) ÷ Max Demand - Normalized gap
        
        **Classification (Improved Ratio):**
        - **< -0.99**: Significantly Overserved (major excess capacity)
        - **-0.99 to -0.1**: Slightly Overserved (some excess)
        - **-0.1 to +0.1**: Well Balanced (reasonable match)
        - **+0.1 to +0.3**: Slightly Underserved (some shortage)
        - **> +0.3**: Significantly Underserved (major shortage)
        
        **Key Improvements:**
        - More realistic load factor ({int(load_factor*100)}% vs 50%)
        - Adjustable baseline to fine-tune sensitivity
        - Multiple calculation methods for different perspectives
        - Improved classification thresholds
        - Better handling of edge cases
        """)

    with st.expander("View Raw Data"):
        display_columns = ['time_slot', 'lat_wgs84', 'lon_wgs84', 'total_passenger', 
                          'jeep_count', 'alley_capacity', 'max_boarding_capacity', 
                          'effective_jeep_arrivals', 'passenger_supply', 'adjusted_supply', 'imbalance_ratio']
        available_columns = [col for col in display_columns if col in filtered_df.columns]
        st.dataframe(filtered_df[available_columns].head(100))

    if st.button("Download Filtered Data as CSV"):
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"passenger_data_{selected_time_slot.replace(' ', '_').lower()}.csv",
            mime="text/csv"
        )

    st.markdown("---")
    st.markdown("*Passenger Density Analysis Dashboard - Built with Streamlit*")

else:
    st.error("Unable to load data. Please check that 'cleaned_data.csv' exists in the same directory as this app.")
    st.info("Make sure your CSV file contains the following columns: 'longitude', 'latitude', 'total_passenger', 'time_slot', 'jeep_count', 'alley_capacity'")