import streamlit as st

def render_help():
    """Render the help page with documentation and user guides"""
    
    # Help sections in tabs
    tabs = st.tabs(["User Guide", "FAQ", "Algorithms", "Troubleshooting", "Contact Support"])
    
    with tabs[0]:  # User Guide
        st.subheader("User Guide")
        
        st.markdown("""
        ## Getting Started with BuildScan AI
        
        BuildScan AI is a powerful tool for detecting structural defects in buildings using 3D point cloud data.
        This guide will help you navigate the system and get the most out of its features.
        
        ### Basic Workflow
        
        1. **Upload Data**: Start by uploading your point cloud data in one of the supported formats (.ply, .pcd, .xyz, .csv).
        2. **Preprocess Data**: Apply preprocessing operations like downsampling, outlier removal, and normal estimation.
        3. **Detect Defects**: Run one of the defect detection algorithms to identify structural issues.
        4. **Visualize Results**: Explore the detected defects using the interactive 3D visualization.
        5. **Generate Reports**: Create comprehensive reports documenting the findings.
        
        ### Navigation
        
        - **Dashboard**: Overview of system status and recent activity
        - **Upload Data**: Upload and preprocess point cloud data
        - **Defect Detection**: Run defect detection algorithms
        - **Reports**: Generate and export analysis reports
        - **History**: View historical scans and track changes over time
        - **Settings**: Configure system settings and preferences
        """)
        
        with st.expander("Data Requirements"):
            st.markdown("""
            ### Point Cloud Data Requirements
            
            - **Supported Formats**: .ply, .pcd, .xyz, .csv
            - **Minimum Points**: At least 1,000 points for meaningful analysis
            - **Recommended Resolution**: 1-5 cm point spacing for building structures
            - **Coordinate System**: Local or global coordinates are supported
            - **Color Information**: Optional but recommended for better visualization
            - **Normal Vectors**: Can be computed if not present in the data
            
            ### Best Practices for Data Collection
            
            - Ensure complete coverage of the structure
            - Maintain consistent scan resolution
            - Avoid excessive noise in the data
            - Include color information when available
            - Use consistent scanning parameters for historical comparison
            """)
            
        with st.expander("System Requirements"):
            st.markdown("""
            ### Web Application
            
            - **Browser**: Modern web browser (Chrome, Firefox, Safari, Edge)
            - **Internet Connection**: Broadband connection required
            - **Device**: Works on desktop, laptop, tablet
            
            ### Desktop Application
            
            - **Operating System**: Windows 10/11, macOS 10.15+, or Linux
            - **Processor**: Intel Core i5/AMD Ryzen 5 or better
            - **Memory**: 8GB RAM minimum, 16GB recommended
            - **Graphics**: Dedicated GPU recommended for large point clouds
            - **Storage**: 500MB for application, plus storage for point cloud data
            - **Python**: 3.8 or higher with PyQt5 installed
            """)
    
    with tabs[1]:  # FAQ
        st.subheader("Frequently Asked Questions")
        
        with st.expander("What file formats are supported?"):
            st.markdown("""
            BuildScan AI supports the following point cloud formats:
            - .ply (Polygon File Format)
            - .pcd (Point Cloud Data)
            - .xyz (XYZ Point Cloud)
            - .csv (Comma-Separated Values)
            
            Each format has different capabilities for storing additional information:
            - PLY: Can store points, colors, normals, and custom properties
            - PCD: Designed for point clouds, supports various properties
            - XYZ: Simple format with just XYZ coordinates
            - CSV: Flexible but requires consistent structure
            """)
        
        with st.expander("How does the defect detection work?"):
            st.markdown("""
            BuildScan AI uses multiple approaches to detect defects:
            
            1. **Feature-Based Detection**: Extracts geometric features (normals, curvature, etc.) and uses machine learning to classify points as defects or normal structure.
            
            2. **Clustering-Based Detection**: Uses unsupervised learning to group points with similar characteristics, identifying clusters that represent potential defects.
            
            3. **Segmentation-Based Detection**: Divides the point cloud into segments based on geometric properties and identifies segments that deviate from expected patterns.
            
            Each method has strengths for different types of defects and building structures.
            """)
        
        with st.expander("Can I compare scans over time?"):
            st.markdown("""
            Yes, BuildScan AI supports comparative analysis between scans taken at different times. This allows you to:
            
            - Track the progression of known defects
            - Identify new defects that have appeared between scans
            - Quantify changes in structural parameters
            - Generate change maps highlighting areas of significant change
            
            For best results, scans should be registered (aligned) to a common coordinate system.
            """)
        
        with st.expander("What types of defects can be detected?"):
            st.markdown("""
            BuildScan AI can detect various types of structural defects, including:
            
            - Surface cracks
            - Spalling and surface deterioration
            - Structural deformations
            - Settlement issues
            - Water damage
            - Misalignments and tilting
            - Missing elements
            
            The effectiveness of detection depends on the resolution of your point cloud data and the nature of the defect.
            """)
        
        with st.expander("How accurate is the defect detection?"):
            st.markdown("""
            The accuracy of defect detection depends on several factors:
            
            - **Data Quality**: Higher resolution scans provide better detection
            - **Defect Type**: Some defects are more visible in point clouds than others
            - **Algorithm Selection**: Different algorithms work better for different scenarios
            - **Parameter Tuning**: Optimal parameters improve detection accuracy
            
            In controlled tests, BuildScan AI typically achieves:
            - 85-95% precision in defect classification
            - 80-90% recall of known defects
            - Sub-centimeter accuracy in defect localization
            
            Real-world performance may vary based on specific conditions.
            """)
    
    with tabs[2]:  # Algorithms
        st.subheader("Algorithms & Technical Details")
        
        st.markdown("""
        BuildScan AI utilizes advanced algorithms for point cloud processing and defect detection.
        This section provides technical details on the main algorithms used in the system.
        """)
        
        with st.expander("Feature Extraction"):
            st.markdown("""
            ### Geometric Feature Extraction
            
            The system extracts the following features from point clouds:
            
            - **Normal Vectors**: Surface orientation at each point
            - **Curvature**: Local surface curvature
            - **Roughness**: Local surface variation
            - **Verticality**: Angle relative to vertical axis
            - **Eigenfeatures**: Derived from PCA of local neighborhoods
            - **Shape Features**: Sphericity, linearity, planarity
            
            These features are computed using k-nearest neighbors or radius-based neighborhood definitions.
            """)
        
        with st.expander("Machine Learning Models"):
            st.markdown("""
            ### Supervised Learning Models
            
            For feature-based defect detection, the following models are available:
            
            - **Random Forest**: Ensemble of decision trees
              - Number of trees: 100
              - Max depth: 20
              - Feature selection: Gini impurity
            
            - **Support Vector Machine (SVM)**:
              - Kernel: Radial Basis Function
              - C parameter: 10.0
              - Gamma: 'scale'
            
            - **Isolation Forest**:
              - Contamination: 'auto'
              - Number of estimators: 100
              
            Models are trained on labeled datasets of building defects or can use transfer learning from pre-trained models.
            """)
        
        with st.expander("Clustering Algorithms"):
            st.markdown("""
            ### Unsupervised Clustering
            
            For clustering-based defect detection:
            
            - **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)**:
              - Works well for irregular shapes
              - Parameters: epsilon (neighborhood size), min_samples
              - Automatically handles noise points
            
            - **K-Means**:
              - Partitions points into k clusters
              - Parameter: number of clusters
              - Works best for globular clusters
            
            - **Hierarchical Clustering**:
              - Builds a hierarchy of clusters
              - Linkage methods: Ward, complete, average
              - Can reveal multi-scale structure
            """)
        
        with st.expander("Segmentation Methods"):
            st.markdown("""
            ### Point Cloud Segmentation
            
            For segmentation-based defect detection:
            
            - **Region Growing**:
              - Grows regions from seed points
              - Criteria: normal consistency, curvature
              - Parameters: smoothness threshold, min cluster size
            
            - **RANSAC (Random Sample Consensus)**:
              - Identifies geometric primitives
              - Models: planes, cylinders, spheres
              - Parameter: distance threshold
            
            - **Plane Fitting**:
              - Fits planes to building components
              - Identifies deviations from expected planes
              - Good for flat structural elements
            """)
        
        with st.expander("Registration & Comparison"):
            st.markdown("""
            ### Multi-temporal Analysis
            
            For comparing scans over time:
            
            - **Point Cloud Registration**:
              - ICP (Iterative Closest Point) algorithm
              - Feature-based matching
              - Global registration followed by fine alignment
            
            - **Change Detection**:
              - Point-to-point distance computation
              - Signed distance fields
              - M3C2 algorithm for directed distances
            
            - **Difference Analysis**:
              - Statistical analysis of distances
              - Threshold-based classification
              - Temporal trend analysis
            """)
    
    with tabs[3]:  # Troubleshooting
        st.subheader("Troubleshooting")
        
        with st.expander("Data Upload Issues"):
            st.markdown("""
            ### Common Data Upload Problems
            
            - **File Too Large**: Try downsampling your point cloud before uploading
            - **Format Error**: Ensure your file is in one of the supported formats (.ply, .pcd, .xyz, .csv)
            - **Corrupt File**: Check if the file can be opened in other point cloud software
            - **Missing Properties**: Ensure required columns/properties are present in your data
            
            ### Solutions
            
            1. For large files: Use the desktop application or downsample the data
            2. For format issues: Convert your file using an external tool like CloudCompare
            3. For corrupt files: Re-export from your scanning software
            4. For missing properties: Add the required columns or use a format that supports the needed properties
            """)
        
        with st.expander("Processing Errors"):
            st.markdown("""
            ### Common Processing Problems
            
            - **Out of Memory**: Point cloud too large for available memory
            - **Processing Timeout**: Operation taking too long
            - **Normal Estimation Failure**: Issues with computing surface normals
            - **Feature Extraction Errors**: Problems with geometric features
            
            ### Solutions
            
            1. For memory issues: Reduce point cloud size or use more aggressive downsampling
            2. For timeouts: Try processing in smaller chunks or use the desktop application
            3. For normal estimation: Adjust the search radius parameter
            4. For feature extraction: Ensure your point cloud has sufficient density for feature computation
            """)
        
        with st.expander("Detection Accuracy Issues"):
            st.markdown("""
            ### Improving Detection Results
            
            - **Too Many False Positives**: System detecting too many defects
            - **Missing Defects**: System not detecting known defects
            - **Incorrect Classification**: Wrong defect types being identified
            
            ### Solutions
            
            1. For false positives: Increase confidence threshold or adjust algorithm parameters
            2. For missing defects: Try a different detection algorithm or reduce the confidence threshold
            3. For misclassification: Use a different feature set or machine learning model
            
            Remember that detection accuracy depends greatly on the quality and resolution of your input data.
            """)
        
        with st.expander("Visualization Problems"):
            st.markdown("""
            ### Common Visualization Issues
            
            - **Slow Rendering**: Point cloud too large for smooth visualization
            - **Color Issues**: Incorrect or missing colors
            - **Display Glitches**: Graphics-related problems
            - **Browser Performance**: Web browser struggling with 3D visualization
            
            ### Solutions
            
            1. For slow rendering: Reduce point size or use downsampling for visualization
            2. For color issues: Check if color information is present in your data
            3. For display glitches: Try a different browser or update your graphics drivers
            4. For browser performance: Close other tabs/applications or try the desktop version
            """)
    
    with tabs[4]:  # Contact Support
        st.subheader("Contact Support")
        
        st.markdown("""
        ### Need Help?
        
        If you're experiencing issues that aren't resolved by the troubleshooting guides,
        please contact our support team:
        
        **Email**: support@buildscan-ai.example.com  
        **Phone**: +1 (555) 123-4567  
        **Hours**: Monday-Friday, 9am-5pm EST
        
        ### Reporting Bugs
        
        To report a bug or issue, please provide the following information:
        
        1. Detailed description of the problem
        2. Steps to reproduce the issue
        3. System information (browser/OS)
        4. Screenshots if applicable
        5. Example data file (if possible)
        
        ### Feature Requests
        
        We're constantly improving BuildScan AI based on user feedback.
        To suggest a new feature, please email us with:
        
        1. Description of the proposed feature
        2. Your use case and why this feature would be valuable
        3. Any specific implementation details you have in mind
        """)
        
        # Contact form (for illustration only)
        st.markdown("### Quick Support Form")
        
        issue_type = st.selectbox("Issue Type", ["Technical Problem", "Account Question", "Feature Request", "Other"])
        
        col1, col2 = st.columns(2)
        
        with col1:
            name = st.text_input("Your Name")
            email = st.text_input("Email Address")
        
        with col2:
            subject = st.text_input("Subject")
            priority = st.select_slider("Priority", options=["Low", "Medium", "High"])
        
        message = st.text_area("Message", height=150)
        
        if st.button("Submit", use_container_width=True):
            st.success("Thank you for your message. Our support team will contact you within 24 hours.")
            st.info("Note: This is a demo form and does not actually submit data in this environment.")