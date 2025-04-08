import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import time

# Import custom modules
import preprocessing
import defect_detection
import visualization
import utils
from help import render_help
from settings import render_settings

def render_home_page():
    """Render the home page with navigation buttons to other pages"""
    st.title("BuildScan AI: 3D Point Cloud-Based Defect Detection")
    
    st.markdown("""
    <div style="display: flex; align-items: center; justify-content: center; margin-bottom: 20px;">
        <img src="https://cdn-icons-png.flaticon.com/512/8291/8291087.png" width="100" height="100" style="margin-right: 20px;">
        <div>
            <h2>Advanced Structural Analysis System</h2>
            <p>Detect building defects automatically using 3D point cloud data and machine learning</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Main navigation buttons
    st.subheader("Start analyzing your building structure")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("📊 Dashboard", use_container_width=True, key="home_dashboard_btn"):
            st.session_state.page = "Dashboard"
            st.rerun()
    
    with col2:
        if st.button("📥 Upload Data", use_container_width=True, key="home_upload_btn"):
            st.session_state.page = "Upload"
            st.rerun()
    
    with col3:
        if st.button("🔍 Defect Detection", use_container_width=True, key="home_detect_btn"):
            st.session_state.page = "Detect"
            st.rerun()
    
    with col4:
        if st.button("📈 Reports", use_container_width=True, key="home_reports_btn"):
            st.session_state.page = "Reports"
            st.rerun()
    
    # Feature Cards
    st.markdown("<h3>Key Capabilities</h3>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="card feature-card">
            <div class="feature-icon">📊</div>
            <div class="feature-title">Advanced Analytics</div>
            <div class="feature-description">Process 3D point cloud data with advanced analytical algorithms to detect structural anomalies and potential defects.</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="card feature-card">
            <div class="feature-icon">🔍</div>
            <div class="feature-title">ML-Based Detection</div>
            <div class="feature-description">Utilize machine learning models to automatically identify and classify different types of structural defects.</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="card feature-card">
            <div class="feature-icon">📈</div>
            <div class="feature-title">Historical Tracking</div>
            <div class="feature-description">Monitor building condition over time with comparative analysis between current and previous scans.</div>
        </div>
        """, unsafe_allow_html=True)
    
    # System Architecture
    st.markdown("<h3>System Architecture</h3>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class="card">
        <p>BuildScan AI is available in two platforms:</p>
        <ul>
            <li><strong>Web Application:</strong> Built with Streamlit, accessible through any modern browser</li>
            <li><strong>Desktop Application:</strong> Built with PyQt5, providing enhanced performance and additional features</li>
        </ul>
        <p>Both versions share the same core functionality:</p>
        <ul>
            <li>Point cloud data processing</li>
            <li>Feature extraction for defect detection</li>
            <li>Multiple detection algorithms (feature-based, clustering, segmentation)</li>
            <li>3D visualization of results</li>
            <li>Report generation</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

def render_dashboard_page():
    """Render the dashboard page with system overview and stats"""
    st.title("Dashboard")
    
    # Return to home button
    if st.button("← Back to Home", key="dashboard_back_btn"):
        st.session_state.page = "Home"
        st.rerun()
    
    # Status Cards
    st.subheader("System Status")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="card">
            <h4>Point Clouds</h4>
            <div style="font-size: 24px; font-weight: bold; color: #3498db;">12</div>
            <p>Total scans in database</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("""
        <div class="card">
            <h4>Defects Detected</h4>
            <div style="font-size: 24px; font-weight: bold; color: #e74c3c;">156</div>
            <p>Across all scans</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown("""
        <div class="card">
            <h4>Buildings</h4>
            <div style="font-size: 24px; font-weight: bold; color: #2ecc71;">5</div>
            <p>Under monitoring</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col4:
        st.markdown("""
        <div class="card">
            <h4>Reports</h4>
            <div style="font-size: 24px; font-weight: bold; color: #f39c12;">8</div>
            <p>Generated this month</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Recent activity
    st.subheader("Recent Activity")
    
    activity_data = pd.DataFrame({
        'Date': ['2025-03-27', '2025-03-26', '2025-03-25', '2025-03-24', '2025-03-23'],
        'Building': ['Office Tower A', 'Residential Complex B', 'Office Tower A', 'Residential Complex B', 'Commercial Center C'],
        'Activity': ['Defect Detection', 'New Scan Upload', 'Report Generated', 'Comparative Analysis', 'New Scan Upload'],
        'User': ['John Smith', 'Emma Watson', 'John Smith', 'David Miller', 'Emma Watson'],
        'Status': ['Completed', 'Completed', 'Completed', 'Completed', 'Completed']
    })
    
    st.dataframe(activity_data, use_container_width=True)
    
    # Actions
    st.subheader("Quick Actions")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("Upload New Scan", use_container_width=True, key="dashboard_upload_btn"):
            st.session_state.page = "Upload"
            st.rerun()
            
    with col2:
        if st.button("Detect Defects", use_container_width=True, key="dashboard_detect_btn"):
            st.session_state.page = "Detect"
            st.rerun()
            
    with col3:
        if st.button("Generate Report", use_container_width=True, key="dashboard_report_btn"):
            st.session_state.page = "Reports"
            st.rerun()
            
    with col4:
        if st.button("View History", use_container_width=True, key="dashboard_history_btn"):
            st.session_state.page = "History"
            st.rerun()

def render_upload_page():
    """Render the data upload and preprocessing page"""
    st.title("Upload Point Cloud Data")
    
    # Return to home button
    if st.button("← Back to Home", key="upload_back_btn"):
        st.session_state.page = "Home"
        st.rerun()
    
    st.markdown("""
    Upload your point cloud data for analysis. The system supports .ply, .pcd, .xyz, and .csv file formats.
    Once uploaded, you can preprocess the data to prepare it for defect detection.
    """)
    
    # File upload section
    uploaded_file = st.file_uploader("Choose a point cloud file", type=["ply", "pcd", "xyz", "csv"])
    
    if uploaded_file:
        st.success(f"File '{uploaded_file.name}' uploaded successfully!")
        
        # Generate a sample point cloud for demo purposes
        with st.spinner("Loading point cloud..."):
            time.sleep(1.5)  # Simulate loading time
            point_cloud, defect_indices = generate_sample_point_cloud()
            st.session_state.point_cloud = point_cloud
        
        # File info
        st.subheader("File Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"**Filename:** {uploaded_file.name}")
            st.markdown(f"**Points:** {len(point_cloud['points']):,}")
            
        with col2:
            st.markdown(f"**Has Colors:** {'Yes' if 'colors' in point_cloud else 'No'}")
            st.markdown(f"**Has Normals:** {'Yes' if 'normals' in point_cloud else 'No'}")
        
        # Preprocessing options
        st.subheader("Preprocessing Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            downsample = st.checkbox("Downsample Point Cloud", value=True)
            if downsample:
                voxel_size = st.slider("Voxel Size", min_value=0.01, max_value=0.5, value=0.05, step=0.01)
            
            remove_outliers = st.checkbox("Remove Outliers", value=True)
            if remove_outliers:
                nb_neighbors = st.slider("Number of Neighbors", min_value=5, max_value=50, value=20, step=5)
                std_ratio = st.slider("Standard Deviation Ratio", min_value=0.5, max_value=5.0, value=2.0, step=0.5)
        
        with col2:
            compute_normals = st.checkbox("Compute Surface Normals", value=True)
            if compute_normals:
                search_radius = st.slider("Search Radius", min_value=0.05, max_value=0.5, value=0.1, step=0.05)
            
            st.markdown("### Additional Options")
            crop_points = st.checkbox("Crop Point Cloud Region", value=False)
            apply_color_filter = st.checkbox("Apply Color Filtering", value=False)
        
        # Process button
        if st.button("Process Data", type="primary", use_container_width=True):
            with st.spinner("Processing point cloud..."):
                time.sleep(2.5)  # Simulate processing time
                
                # In a real scenario, we would call preprocessing.preprocess_point_cloud() with the selected options
                st.session_state.processed_point_cloud = point_cloud  # For demo, just use the original
                
                # Display a preview
                fig = visualization.visualize_point_cloud(point_cloud, point_size=2)
                st.plotly_chart(fig, use_container_width=True)
                
                st.success("Point cloud processed successfully!")
                
                # Continue to detection button
                if st.button("Continue to Defect Detection", use_container_width=True):
                    st.session_state.page = "Detect"
                    st.rerun()
    else:
        # Demo option if no file is uploaded
        st.info("No file uploaded. You can use a demo point cloud instead.")
        if st.button("Use Demo Point Cloud", use_container_width=True):
            with st.spinner("Generating demo point cloud..."):
                time.sleep(1.5)  # Simulate loading time
                point_cloud, defect_indices = generate_sample_point_cloud()
                st.session_state.point_cloud = point_cloud
                st.session_state.processed_point_cloud = point_cloud
                
                # Continue to detection button
                st.success("Demo point cloud loaded!")
                if st.button("Continue to Defect Detection", use_container_width=True):
                    st.session_state.page = "Detect"
                    st.rerun()

def render_detection_page():
    """Render the defect detection page"""
    st.title("Defect Detection")
    
    # Return to home button
    if st.button("← Back to Home", key="detect_back_btn"):
        st.session_state.page = "Home"
        st.rerun()
    
    # Check if we have a point cloud loaded
    if st.session_state.processed_point_cloud is None:
        st.warning("No point cloud data available. Please upload and process data first.")
        if st.button("Go to Upload Page", use_container_width=True):
            st.session_state.page = "Upload"
            st.rerun()
            
        # Option to use demo data
        st.info("Alternatively, you can use a demo point cloud.")
        if st.button("Use Demo Point Cloud", use_container_width=True):
            with st.spinner("Generating demo point cloud..."):
                time.sleep(1.5)  # Simulate loading time
                point_cloud, defect_indices = generate_sample_point_cloud()
                st.session_state.point_cloud = point_cloud
                st.session_state.processed_point_cloud = point_cloud
                st.rerun()
    else:
        st.markdown("""
        Select a detection method and configure its parameters to identify structural defects in the point cloud.
        The system supports multiple detection approaches optimized for different types of defects.
        """)
        
        # Detection method selection
        detection_method = st.selectbox(
            "Detection Method",
            ["Feature-Based Detection", "Clustering-Based Detection", "Segmentation-Based Detection"]
        )
        
        # Parameters based on selected method
        if detection_method == "Feature-Based Detection":
            col1, col2 = st.columns(2)
            
            with col1:
                model_type = st.selectbox(
                    "Machine Learning Model",
                    ["Random Forest", "SVM", "Isolation Forest"]
                )
                
                feature_set = st.multiselect(
                    "Features to Use",
                    ["Normals", "Curvature", "Roughness", "Verticality"],
                    default=["Normals", "Curvature", "Roughness"]
                )
            
            with col2:
                confidence_threshold = st.slider(
                    "Confidence Threshold", 
                    min_value=0.0, 
                    max_value=1.0, 
                    value=0.7, 
                    step=0.05
                )
                
                use_ensemble = st.checkbox("Use Ensemble Methods", value=True)
        
        elif detection_method == "Clustering-Based Detection":
            col1, col2 = st.columns(2)
            
            with col1:
                clustering_method = st.selectbox(
                    "Clustering Algorithm",
                    ["DBSCAN", "K-Means", "Hierarchical"]
                )
                
                if clustering_method == "DBSCAN":
                    eps = st.slider("Epsilon (Neighborhood Size)", min_value=0.05, max_value=1.0, value=0.1, step=0.05)
                    min_samples = st.slider("Minimum Samples", min_value=5, max_value=50, value=10, step=5)
                elif clustering_method == "K-Means":
                    n_clusters = st.slider("Number of Clusters", min_value=2, max_value=20, value=5, step=1)
            
            with col2:
                feature_set = st.multiselect(
                    "Features for Clustering",
                    ["Spatial Coordinates", "Colors", "Normals", "Curvature"],
                    default=["Spatial Coordinates", "Normals"]
                )
                
                outlier_threshold = st.slider(
                    "Outlier Threshold", 
                    min_value=1.0, 
                    max_value=10.0, 
                    value=3.0, 
                    step=0.5
                )
        
        elif detection_method == "Segmentation-Based Detection":
            col1, col2 = st.columns(2)
            
            with col1:
                segmentation_method = st.selectbox(
                    "Segmentation Method",
                    ["Region Growing", "RANSAC", "Plane Fitting"]
                )
                
                if segmentation_method == "Region Growing":
                    smoothness_threshold = st.slider("Smoothness Threshold", min_value=1.0, max_value=10.0, value=3.0, step=0.5)
                    min_cluster_size = st.slider("Minimum Cluster Size", min_value=50, max_value=500, value=100, step=50)
                elif segmentation_method == "RANSAC":
                    distance_threshold = st.slider("Distance Threshold", min_value=0.01, max_value=0.2, value=0.05, step=0.01)
            
            with col2:
                normal_consistency = st.checkbox("Use Normal Consistency", value=True)
                merge_similar_regions = st.checkbox("Merge Similar Regions", value=True)
                
                deviation_threshold = st.slider(
                    "Deviation Threshold", 
                    min_value=0.01, 
                    max_value=0.5, 
                    value=0.1, 
                    step=0.01
                )
        
        # Run detection button
        if st.button("Run Defect Detection", type="primary", use_container_width=True):
            with st.spinner("Detecting defects..."):
                time.sleep(3.0)  # Simulate detection time
                
                # In a real scenario, we would call appropriate detection function based on selected method
                # For demo, generate random defect indices
                point_cloud = st.session_state.processed_point_cloud
                defect_mask = np.random.choice([False, True], len(point_cloud['points']), p=[0.9, 0.1])
                defect_indices = np.where(defect_mask)[0]
                defect_labels = np.zeros(len(point_cloud['points']))
                defect_labels[defect_indices] = 1
                
                st.session_state.defect_indices = defect_indices
                st.session_state.defect_labels = defect_labels
                
                # Show results
                st.success(f"Detection complete! Found {len(defect_indices)} potential defects.")
                
                # Visualization
                st.subheader("Defect Visualization")
                
                # Create a copy of the point cloud with highlighted defects
                vis_point_cloud = point_cloud.copy()
                colors = vis_point_cloud['colors'].copy()
                colors[defect_indices] = np.array([1.0, 0.0, 0.0])  # Red for defects
                vis_point_cloud['colors'] = colors
                
                fig = visualization.visualize_point_cloud(vis_point_cloud, point_size=3)
                st.plotly_chart(fig, use_container_width=True)
                
                # Statistics
                st.subheader("Defect Statistics")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Points", f"{len(point_cloud['points']):,}")
                    
                with col2:
                    st.metric("Defective Points", f"{len(defect_indices):,}")
                    
                with col3:
                    defect_percentage = (len(defect_indices) / len(point_cloud['points'])) * 100
                    st.metric("Defect Percentage", f"{defect_percentage:.2f}%")
                
                # Classification report (dummy for demo)
                st.subheader("Classification Report")
                
                report_data = pd.DataFrame({
                    'Metric': ['Precision', 'Recall', 'F1-Score', 'Accuracy'],
                    'Score': [0.89, 0.92, 0.90, 0.95]
                })
                
                st.dataframe(report_data, use_container_width=True)
                
                # Actions
                st.subheader("Next Steps")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("Generate Report", use_container_width=True):
                        st.session_state.page = "Reports"
                        st.rerun()
                
                with col2:
                    if st.button("Export Results", use_container_width=True):
                        st.info("In a production environment, this would export the results to various formats (CSV, PLY, etc.).")

def render_reports_page():
    """Render the reports generation page"""
    st.title("Report Generation")
    
    # Return to home button
    if st.button("← Back to Home", key="reports_back_btn"):
        st.session_state.page = "Home"
        st.rerun()
    
    # Check if we have detection results
    if st.session_state.defect_indices is None:
        st.warning("No defect detection results available. Please run defect detection first.")
        if st.button("Go to Detection Page", use_container_width=True):
            st.session_state.page = "Detect"
            st.rerun()
        
        # Option to use demo data
        st.info("Alternatively, you can use demo results.")
        if st.button("Use Demo Results", use_container_width=True):
            with st.spinner("Generating demo results..."):
                time.sleep(1.5)  # Simulate loading time
                
                point_cloud, defect_indices = generate_sample_point_cloud()
                st.session_state.point_cloud = point_cloud
                st.session_state.processed_point_cloud = point_cloud
                
                defect_labels = np.zeros(len(point_cloud['points']))
                defect_labels[defect_indices] = 1
                
                st.session_state.defect_indices = defect_indices
                st.session_state.defect_labels = defect_labels
                
                st.rerun()
    else:
        st.markdown("""
        Generate comprehensive reports based on the defect detection results.
        These reports can be shared with stakeholders, archived for historical reference,
        or used for planning remediation activities.
        """)
        
        # Report configuration
        st.subheader("Report Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            report_title = st.text_input("Report Title", "Building Structure Defect Analysis Report")
            include_visualizations = st.checkbox("Include Visualizations", value=True)
            include_statistics = st.checkbox("Include Statistics", value=True)
            include_raw_data = st.checkbox("Include Raw Data Tables", value=False)
        
        with col2:
            report_format = st.selectbox("Report Format", ["HTML", "PDF", "CSV", "JSON"])
            include_recommendations = st.checkbox("Include Recommendations", value=True)
            include_executive_summary = st.checkbox("Include Executive Summary", value=True)
            include_methodology = st.checkbox("Include Methodology Description", value=True)
        
        # Building information
        st.subheader("Building Information")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            building_name = st.text_input("Building Name", "Tower A")
            building_location = st.text_input("Location", "123 Main Street")
        
        with col2:
            scan_date = st.date_input("Scan Date")
            scan_equipment = st.text_input("Scan Equipment", "Leica BLK360")
        
        with col3:
            report_author = st.text_input("Report Author", "John Smith")
            organization = st.text_input("Organization", "BuildScan AI")
        
        # Generate report button
        if st.button("Generate Report", type="primary", use_container_width=True):
            with st.spinner("Generating report..."):
                time.sleep(3.0)  # Simulate report generation time
                
                # In a real scenario, we would call reporting.generate_report() with the selected options
                
                # Show a preview of the report
                st.success("Report generated successfully!")
                
                # Report preview
                st.subheader("Report Preview")
                
                tab1, tab2, tab3 = st.tabs(["Summary", "Visualizations", "Statistics"])
                
                with tab1:
                    st.markdown(f"""
                    # {report_title}
                    
                    **Building:** {building_name}  
                    **Location:** {building_location}  
                    **Date of Scan:** {scan_date}  
                    **Report Author:** {report_author}  
                    **Organization:** {organization}
                    
                    ## Executive Summary
                    
                    This report presents the findings of a structural defect analysis conducted on {building_name}
                    using 3D point cloud data processed by BuildScan AI's advanced defect detection algorithms.
                    
                    The analysis identified **{len(st.session_state.defect_indices)}** potential defects in the structure,
                    representing approximately **{(len(st.session_state.defect_indices) / len(st.session_state.processed_point_cloud['points'])) * 100:.2f}%**
                    of the total scanned surface area.
                    
                    ### Key Findings
                    
                    - Multiple small cracks detected in the east façade
                    - Surface deterioration observed on the north-facing exterior walls
                    - Minor structural deformation detected in the roof structure
                    - Water damage identified in the basement level
                    
                    ### Recommendations
                    
                    1. Conduct detailed inspection of the east façade cracks
                    2. Schedule repair of the surface deterioration within 3-6 months
                    3. Monitor the roof deformation quarterly
                    4. Address water infiltration sources in the basement immediately
                    
                    """)
                
                with tab2:
                    # Point cloud visualization with defects highlighted
                    point_cloud = st.session_state.processed_point_cloud
                    defect_indices = st.session_state.defect_indices
                    
                    # Create a copy of the point cloud with highlighted defects
                    vis_point_cloud = point_cloud.copy()
                    colors = vis_point_cloud['colors'].copy()
                    colors[defect_indices] = np.array([1.0, 0.0, 0.0])  # Red for defects
                    vis_point_cloud['colors'] = colors
                    
                    fig = visualization.visualize_point_cloud(vis_point_cloud, point_size=3)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Defect distribution visualization
                    defect_distribution = pd.DataFrame({
                        'Type': ['Cracks', 'Surface Deterioration', 'Structural Deformation', 'Water Damage'],
                        'Count': [42, 35, 18, 26],
                        'Severity': ['High', 'Medium', 'Low', 'Medium']
                    })
                    
                    fig = px.bar(
                        defect_distribution, 
                        x='Type', 
                        y='Count',
                        color='Severity',
                        color_discrete_map={
                            'High': '#e74c3c',
                            'Medium': '#f39c12',
                            'Low': '#3498db'
                        },
                        title="Defect Distribution by Type and Severity"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                with tab3:
                    # Defect statistics
                    st.markdown("### General Statistics")
                    
                    stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
                    
                    with stats_col1:
                        st.metric("Total Points", f"{len(point_cloud['points']):,}")
                    
                    with stats_col2:
                        st.metric("Defective Points", f"{len(defect_indices):,}")
                    
                    with stats_col3:
                        defect_percentage = (len(defect_indices) / len(point_cloud['points'])) * 100
                        st.metric("Defect Percentage", f"{defect_percentage:.2f}%")
                    
                    with stats_col4:
                        st.metric("Defect Types", "4")
                    
                    # Defect distribution table
                    st.markdown("### Defect Distribution")
                    
                    defect_distribution = pd.DataFrame({
                        'Type': ['Cracks', 'Surface Deterioration', 'Structural Deformation', 'Water Damage'],
                        'Count': [42, 35, 18, 26],
                        'Percentage': ['34.7%', '28.9%', '14.9%', '21.5%'],
                        'Severity': ['High', 'Medium', 'Low', 'Medium'],
                        'Priority': ['Immediate', 'Medium', 'Low', 'High']
                    })
                    
                    st.dataframe(defect_distribution, use_container_width=True)
                    
                    # Classification metrics
                    st.markdown("### Classification Metrics")
                    
                    metrics_data = pd.DataFrame({
                        'Metric': ['Precision', 'Recall', 'F1-Score', 'Accuracy'],
                        'Score': [0.89, 0.92, 0.90, 0.95]
                    })
                    
                    st.dataframe(metrics_data, use_container_width=True)
                
                # Export options
                st.subheader("Export Options")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button("Download HTML Report", use_container_width=True):
                        st.info("In a production environment, this would download the HTML report.")
                
                with col2:
                    if st.button("Download PDF Report", use_container_width=True):
                        st.info("In a production environment, this would download the PDF report.")
                
                with col3:
                    if st.button("Export Raw Data", use_container_width=True):
                        st.info("In a production environment, this would export the raw defect data.")

def render_history_page():
    """Render the history page to view previous scans"""
    st.title("Scan History")
    
    # Return to home button
    if st.button("← Back to Home", key="history_back_btn"):
        st.session_state.page = "Home"
        st.rerun()
    
    st.markdown("""
    View the history of building scans and defect detection results over time.
    This allows you to track changes in building condition and monitor defect progression.
    """)
    
    # Building selector
    building = st.selectbox("Select Building", ["All Buildings", "Tower A", "Tower B", "Residential Complex C", "Commercial Center D"])
    
    # Date range
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=pd.to_datetime("2024-01-01"))
    with col2:
        end_date = st.date_input("End Date")
    
    # Filter options
    with st.expander("Advanced Filters"):
        col1, col2 = st.columns(2)
        
        with col1:
            scan_type = st.multiselect("Scan Type", ["Full Building", "Exterior Only", "Interior Only", "Specific Floor"], default=["Full Building"])
            defect_types = st.multiselect("Defect Types", ["Cracks", "Surface Deterioration", "Structural Deformation", "Water Damage"], default=["Cracks", "Surface Deterioration", "Structural Deformation", "Water Damage"])
        
        with col2:
            severity = st.multiselect("Severity", ["High", "Medium", "Low"], default=["High", "Medium", "Low"])
            status = st.multiselect("Status", ["New", "Monitoring", "Scheduled for Repair", "Repaired"], default=["New", "Monitoring", "Scheduled for Repair"])
    
    # Apply filters button
    st.button("Apply Filters", use_container_width=True)
    
    # History table
    st.subheader("Scan History")
    
    # Sample data for demonstration
    history_data = pd.DataFrame({
        'Date': ['2025-03-15', '2025-02-10', '2025-01-05', '2024-12-01', '2024-11-01', '2024-10-01'],
        'Building': ['Tower A', 'Tower A', 'Tower A', 'Tower A', 'Tower A', 'Tower A'],
        'Scan Type': ['Full Building', 'Full Building', 'Exterior Only', 'Full Building', 'Full Building', 'Full Building'],
        'Points': ['1.2M', '1.2M', '650K', '1.1M', '1.2M', '1.2M'],
        'Defects': [121, 115, 68, 110, 95, 80],
        'Change': ['+5.2%', '+6.5%', '-', '+15.8%', '+18.8%', '-'],
        'Status': ['New', 'Monitoring', 'Monitoring', 'Monitoring', 'Monitoring', 'Monitoring']
    })
    
    st.dataframe(history_data, use_container_width=True)
    
    # Select scan to view details
    st.subheader("Select Scan to View Details")
    
    selected_scan = st.selectbox("Select Scan", history_data['Date'] + ' - ' + history_data['Scan Type'])
    
    if selected_scan:
        st.subheader("Scan Details")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Mock point cloud visualization
            st.markdown("### Point Cloud")
            st.image("https://cdn-icons-png.flaticon.com/512/8291/8291087.png", width=200)
            
            st.markdown("### Defect Distribution")
            # Mock chart
            defect_distribution = pd.DataFrame({
                'Type': ['Cracks', 'Surface Deterioration', 'Structural Deformation', 'Water Damage'],
                'Count': [42, 35, 18, 26]
            })
            
            fig = px.pie(
                defect_distribution, 
                values='Count', 
                names='Type',
                title="Defect Distribution",
                color_discrete_sequence=px.colors.qualitative.Bold
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Scan details
            st.markdown("### Scan Information")
            
            st.markdown(f"""
            **Date:** {selected_scan.split(' - ')[0]}  
            **Scan Type:** {selected_scan.split(' - ')[1]}  
            **Building:** Tower A  
            **Operator:** John Smith  
            **Equipment:** Leica BLK360  
            **Processing Time:** 45 minutes  
            """)
            
            st.markdown("### Defect Summary")
            
            defect_summary = pd.DataFrame({
                'Type': ['Cracks', 'Surface Deterioration', 'Structural Deformation', 'Water Damage'],
                'Count': [42, 35, 18, 26],
                'Severity': ['High', 'Medium', 'Low', 'Medium']
            })
            
            st.dataframe(defect_summary, use_container_width=True)
            
            st.markdown("### Actions")
            
            if st.button("View Full Report", use_container_width=True):
                st.info("In a production environment, this would open the full report for the selected scan.")
            
            if st.button("Compare with Latest Scan", use_container_width=True):
                st.info("In a production environment, this would initiate a comparison between this scan and the latest scan.")
            
            if st.button("Export Data", use_container_width=True):
                st.info("In a production environment, this would export the data for the selected scan.")

def render_settings_page():
    """Render the settings page"""
    st.title("Settings")
    
    # Return to home button
    if st.button("← Back to Home", key="settings_back_btn"):
        st.session_state.page = "Home"
        st.rerun()
    
    # Call the settings module's render function
    render_settings()

def render_help_page():
    """Render the help page"""
    st.title("Help & Documentation")
    
    # Return to home button
    if st.button("← Back to Home", key="help_back_btn"):
        st.session_state.page = "Home"
        st.rerun()
    
    # Call the help module's render function
    render_help()

# Function to generate a sample point cloud for demo
def generate_sample_point_cloud(num_points=5000):
    """Generate a sample point cloud with some artificial defects"""
    # Generate regular building structure points (cube-like shape)
    x = np.random.uniform(-5, 5, num_points)
    y = np.random.uniform(-5, 5, num_points)
    z = np.random.uniform(0, 10, num_points)
    
    # Basic colors (gray for structure)
    colors = np.ones((num_points, 3)) * 0.7  # Gray color
    
    # Add some defects (random points with different colors)
    defect_mask = np.random.choice([False, True], num_points, p=[0.9, 0.1])
    defect_indices = np.where(defect_mask)[0]
    
    # Color defects in red
    colors[defect_indices] = np.array([1.0, 0.0, 0.0])  # Red color for defects
    
    # Create normals (random for simplicity)
    normals = np.random.uniform(-1, 1, (num_points, 3))
    normals = normals / np.linalg.norm(normals, axis=1)[:, np.newaxis]
    
    # Package as a point cloud dictionary
    return {
        'points': np.column_stack((x, y, z)),
        'colors': colors,
        'normals': normals
    }, defect_indices