import streamlit as st
import requests
from PIL import Image
import io

# Page configuration
st.set_page_config(
    page_title="Car Damage Severity Detection",
    page_icon="🚗",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom styling
st.markdown("""
    <style>
    .main {
        padding-top: 0rem;
    }
    .stTitle {
        text-align: center;
        color: #1f77b4;
    }
    </style>
    """, unsafe_allow_html=True)

# Header
st.markdown("## 🚗 Car Damage Severity Detection")
st.markdown("---")
st.write("Upload a car image to analyze its damage severity level.")

# Create columns for better layout
col1, col2 = st.columns([1, 1], gap="medium")

with col1:
    st.subheader("📸 Upload Image")
    file = st.file_uploader("Choose a car image", type=["jpg", "jpeg", "png", "bmp"])

with col2:
    st.subheader("📊 Results")
    result_placeholder = st.empty()

if file:
    # Display the image
    st.markdown("---")
    st.subheader("Preview")
    st.image(file, width='stretch')
    
    # Show processing status
    with st.spinner("🔍 Analyzing image... Please wait"):
        try:
            # Send to backend
            response = requests.post(
                "http://localhost:8000/predict",
                files={"file": file.getvalue()}
            )
            
            if response.status_code == 200:
                result = response.json()
                
                if "error" not in result:
                    # Display results with nice formatting
                    st.markdown("---")
                    st.subheader("✅ Analysis Complete")
                    
                    # Create columns for metrics
                    metric_col1, metric_col2 = st.columns(2)
                    
                    with metric_col1:
                        prediction = result.get("prediction", "Unknown").upper()
                        st.metric("Severity Level", prediction)
                    
                    # Color-coded severity message
                    severity_colors = {
                        "minor": "🟢 Minor damage detected - Small repairs needed",
                        "moderate": "🟡 Moderate damage detected - Medium repairs needed",
                        "severe": "🔴 Severe damage detected - Major repairs needed"
                    }
                    
                    severity_msg = severity_colors.get(prediction.lower(), "Unknown damage level")
                    st.info(severity_msg)
                else:
                    st.error(f"Error: {result['error']}")
            else:
                st.error("❌ Failed to connect to the backend server. Make sure it's running on port 8000.")
        except requests.exceptions.ConnectionError:
            st.error("❌ Cannot connect to backend server. Please ensure the backend is running on http://localhost:8000")
        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: gray; font-size: 12px;'>CNN-based Car Damage Severity Detection Model</p>", unsafe_allow_html=True)