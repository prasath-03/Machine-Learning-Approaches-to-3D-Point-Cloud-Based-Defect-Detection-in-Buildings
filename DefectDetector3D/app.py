import streamlit as st
import os
import base64
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
import app_pages

# Must be the first Streamlit command
st.set_page_config(
    page_title="BuildScan AI | 3D Point Cloud-Based Defect Detection",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
with open('static/style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Hide default streamlit elements
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
.stDeployButton {display:none;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Initialize session state for navigation
if 'page' not in st.session_state:
    st.session_state.page = 'Home'

if 'point_cloud' not in st.session_state:
    st.session_state.point_cloud = None

if 'processed_point_cloud' not in st.session_state:
    st.session_state.processed_point_cloud = None
    
if 'defect_indices' not in st.session_state:
    st.session_state.defect_indices = None
    
if 'defect_labels' not in st.session_state:
    st.session_state.defect_labels = None

# Function to load image as base64
def img_to_base64(img_path):
    with open(img_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

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

# Navigation sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/8291/8291087.png", width=100)
    st.title("BuildScan AI")
    
    st.markdown("### Navigation")
    
    if st.button("🏠 Home", use_container_width=True):
        st.session_state.page = "Home"
        st.rerun()
        
    if st.button("📊 Dashboard", use_container_width=True):
        st.session_state.page = "Dashboard"
        st.rerun()
        
    if st.button("📥 Upload Data", use_container_width=True):
        st.session_state.page = "Upload"
        st.rerun()
        
    if st.button("🔍 Defect Detection", use_container_width=True):
        st.session_state.page = "Detect"
        st.rerun()
        
    if st.button("📈 Reports", use_container_width=True):
        st.session_state.page = "Reports"
        st.rerun()
        
    if st.button("📁 History", use_container_width=True):
        st.session_state.page = "History"
        st.rerun()
        
    if st.button("⚙️ Settings", use_container_width=True):
        st.session_state.page = "Settings"
        st.rerun()
        
    if st.button("❓ Help", use_container_width=True):
        st.session_state.page = "Help"
        st.rerun()

# Render the appropriate page based on the session state
if st.session_state.page == "Home":
    app_pages.render_home_page()
elif st.session_state.page == "Dashboard":
    app_pages.render_dashboard_page()
elif st.session_state.page == "Upload":
    app_pages.render_upload_page()
elif st.session_state.page == "Detect":
    app_pages.render_detection_page()
elif st.session_state.page == "Reports":
    app_pages.render_reports_page()
elif st.session_state.page == "History":
    app_pages.render_history_page()
elif st.session_state.page == "Settings":
    app_pages.render_settings_page()
elif st.session_state.page == "Help":
    app_pages.render_help_page()
else:
    # Create tabs for landing page
    st.title("BuildScan AI: 3D Point Cloud-Based Defect Detection")
    tabs = st.tabs(["Overview", "Web Interface", "Desktop Application", "Documentation"])
    
    # Tab 0: Overview
    with tabs[0]:
        st.header("Overview")
    
        st.markdown("""
        <div style="display: flex; align-items: center; justify-content: center; margin-bottom: 20px;">
            <img src="https://cdn-icons-png.flaticon.com/512/8291/8291087.png" width="100" height="100" style="margin-right: 20px;">
            <div>
                <h2>Advanced Structural Analysis System</h2>
                <p>Detect building defects automatically using 3D point cloud data and machine learning</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
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
    
    # Tab 1: Web Interface
    with tabs[1]:
        st.header("Web Interface (Streamlit)")
    
        st.markdown("""
        The Streamlit web interface provides core functionality in an easy-to-use web application.
        This interface is perfect for users who need to access the system remotely or prefer a browser-based experience.
        """)
    
        # Web Interface Features
        st.subheader("Key Features")
        
        col1, col2 = st.columns(2)
    
        with col1:
            st.markdown("""
            <div class="card">
                <h4>Data Processing</h4>
                <ul>
                    <li>Upload point cloud files (.ply, .pcd, .xyz, .csv)</li>
                    <li>Preprocess data (downsampling, outlier removal, normal estimation)</li>
                    <li>Feature extraction for defect detection</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="card">
                <h4>Analysis Tools</h4>
                <ul>
                    <li>Machine learning-based defect detection</li>
                    <li>Clustering algorithms for anomaly detection</li>
                    <li>Geometric feature analysis</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="card">
                <h4>Visualization</h4>
                <ul>
                    <li>Interactive 3D point cloud viewer</li>
                    <li>Color-coded defect highlighting</li>
                    <li>Statistical charts and graphs</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="card">
                <h4>Reporting</h4>
                <ul>
                    <li>Generate detailed defect reports</li>
                    <li>Export analysis results</li>
                    <li>Historical data comparison</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Sample upload section
        st.subheader("Try It Now")
        upload_file = st.file_uploader("Upload a point cloud file for analysis", type=["ply", "pcd", "xyz", "csv"])
    
        if upload_file is not None:
            st.info("File uploaded. In a production environment, this would trigger the processing pipeline.")
            
            # Show a sample analysis result
            st.subheader("Sample Analysis Result")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("""
                <div style="background-color: #f8f9fa; padding: 20px; border-radius: 10px;">
                    <h4>3D Visualization would appear here</h4>
                    <p style="color: #666;">The actual system would render an interactive 3D view of your point cloud with detected defects highlighted in different colors.</p>
                </div>
                """, unsafe_allow_html=True)
                
            with col2:
                st.markdown("""
                <div class="card">
                    <h4>Statistics</h4>
                    <p><strong>Total points:</strong> 12,458</p>
                    <p><strong>Detected defects:</strong> 47</p>
                    <p><strong>Defect categories:</strong></p>
                    <ul>
                        <li>Cracks: 18</li>
                        <li>Surface deterioration: 12</li>
                        <li>Structural deformation: 9</li>
                        <li>Water damage: 8</li>
                    </ul>
                    <p><strong>Overall condition:</strong> Requires attention</p>
                </div>
                """, unsafe_allow_html=True)
    
    # Tab 2: Desktop Application
    with tabs[2]:
        st.header("Desktop Application (PyQt)")
        
        st.image("generated-icon.png", width=100)
        
        st.markdown("""
        The PyQt-based desktop application provides enhanced performance and additional features.
        It's designed for professionals who need the full range of capabilities and optimal performance for large point cloud datasets.
        
        This application requires a system with PyQt5 installed and cannot run directly in a web browser environment.
        """)
        
        # Desktop app features
        st.subheader("Enhanced Capabilities")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="card feature-card">
                <div class="feature-icon">⚡</div>
                <div class="feature-title">Performance</div>
                <div class="feature-description">Optimized for handling large point cloud datasets with native GPU acceleration support.</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="card feature-card">
                <div class="feature-icon">🔄</div>
                <div class="feature-title">Advanced Workflow</div>
                <div class="feature-description">Complex multi-step analysis workflows with batch processing capabilities.</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="card feature-card">
                <div class="feature-icon">🏢</div>
                <div class="feature-title">Project Management</div>
                <div class="feature-description">Advanced project management for tracking multiple buildings and construction sites.</div>
            </div>
            """, unsafe_allow_html=True)
            
        # Desktop interface preview
        st.subheader("Desktop Interface Preview")
        
        with st.expander("View Screenshot"):
            st.markdown("""
            <div style="text-align: center; background-color: #f8f9fa; padding: 20px; border-radius: 10px;">
                <img src="https://cdn-icons-png.flaticon.com/512/8291/8291087.png" width="200">
                <p>The desktop application provides a comprehensive interface with multiple panels for visualization, analysis, and reporting.</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Download info
        st.subheader("Download Information")
        
        st.markdown("""
        <div class="card">
            <h4>System Requirements</h4>
            <ul>
                <li>Operating System: Windows 10/11, macOS 10.15+, or Linux</li>
                <li>Python 3.8 or higher</li>
                <li>PyQt5 and related dependencies</li>
                <li>GPU recommended for optimal performance</li>
            </ul>
            <p>Please contact your system administrator for installation assistance.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Tab 3: Documentation
    with tabs[3]:
        st.header("Documentation & Resources")
        
        # Quick start guide
        st.subheader("Quick Start Guide")
        
        st.markdown("""
        <div class="card">
            <h4>Getting Started</h4>
            <ol>
                <li><strong>Data Preparation:</strong> Collect 3D point cloud scans of the building structure using LiDAR or photogrammetry</li>
                <li><strong>Data Import:</strong> Upload the point cloud files in supported formats (.ply, .pcd, .xyz, .csv)</li>
                <li><strong>Preprocessing:</strong> Apply downsampling, outlier removal, and normal estimation as needed</li>
                <li><strong>Defect Detection:</strong> Run one of the available detection algorithms (feature-based, clustering, segmentation)</li>
                <li><strong>Analysis:</strong> Review the detected defects, their locations, and severity</li>
                <li><strong>Reporting:</strong> Generate comprehensive reports for documentation and planning</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
        
        # Core algorithms
        st.subheader("Core Algorithms")
        
        tab_alg1, tab_alg2, tab_alg3 = st.tabs(["Feature-Based Detection", "Clustering", "Segmentation"])
        
        with tab_alg1:
            st.markdown("""
            <div class="card">
                <h4>Feature-Based Detection</h4>
                <p>This method extracts geometric features from point clouds and uses supervised machine learning to classify points as defects or normal structure.</p>
                <h5>Key Features:</h5>
                <ul>
                    <li>Normal vectors</li>
                    <li>Surface curvature</li>
                    <li>Roughness</li>
                    <li>Verticality</li>
                </ul>
                <h5>Machine Learning Models:</h5>
                <ul>
                    <li>Random Forest</li>
                    <li>Support Vector Machine (SVM)</li>
                    <li>Isolation Forest</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
        with tab_alg2:
            st.markdown("""
            <div class="card">
                <h4>Clustering-Based Detection</h4>
                <p>This method uses unsupervised learning to group points with similar characteristics, identifying clusters that represent potential defects.</p>
                <h5>Algorithms:</h5>
                <ul>
                    <li>DBSCAN (Density-Based Spatial Clustering of Applications with Noise)</li>
                    <li>K-Means</li>
                    <li>Hierarchical Clustering</li>
                </ul>
                <h5>Key Parameters:</h5>
                <ul>
                    <li>Epsilon (neighborhood size)</li>
                    <li>Minimum cluster size</li>
                    <li>Distance metrics</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
        with tab_alg3:
            st.markdown("""
            <div class="card">
                <h4>Segmentation-Based Detection</h4>
                <p>This method divides the point cloud into segments based on geometric properties and identifies segments that deviate from expected patterns.</p>
                <h5>Techniques:</h5>
                <ul>
                    <li>Region Growing</li>
                    <li>RANSAC (Random Sample Consensus)</li>
                    <li>Plane Fitting</li>
                </ul>
                <h5>Applications:</h5>
                <ul>
                    <li>Detecting cracks in flat surfaces</li>
                    <li>Identifying structural deformations</li>
                    <li>Separating building components for targeted analysis</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Support resources
        st.subheader("Support Resources")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="card">
                <h4>Technical Support</h4>
                <p>For technical assistance with the application, please contact:</p>
                <p><strong>Email:</strong> support@buildscan-ai.example.com</p>
                <p><strong>Phone:</strong> +1 (555) 123-4567</p>
                <p><strong>Hours:</strong> Monday-Friday, 9 AM - 5 PM EST</p>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown("""
            <div class="card">
                <h4>Learning Resources</h4>
                <p>Enhance your knowledge with these resources:</p>
                <ul>
                    <li>User Manual</li>
                    <li>Video Tutorials</li>
                    <li>Case Studies</li>
                    <li>API Documentation</li>
                    <li>Best Practices Guide</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

