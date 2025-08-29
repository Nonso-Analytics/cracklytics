import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
import cv2
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import io
import time
import plotly.graph_objects as go
import plotly.express as px

# Set page config
st.set_page_config(
    page_title="Crack Detection App",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main {
    padding-top: 2rem;
}
.stAlert {
    margin-top: 1rem;
}
.metric-container {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    padding: 1rem;
    border-radius: 10px;
    color: white;
    margin: 0.5rem 0;
}
</style>
""", unsafe_allow_html=True)

# Check device
@st.cache_resource
def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

device = get_device()

# Model definition (same as your training code)
class CrackDetectionModel(nn.Module):
    def __init__(self, model_name='mobilenet_v2', pretrained=True, num_classes=1):
        super(CrackDetectionModel, self).__init__()

        if model_name == 'mobilenet_v2':
            self.backbone = models.mobilenet_v2(pretrained=pretrained)
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(self.backbone.last_channel, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, num_classes)
            )

        elif model_name == 'resnet18':
            self.backbone = models.resnet18(pretrained=pretrained)
            self.backbone.fc = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(self.backbone.fc.in_features, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, num_classes)
            )

        elif model_name == 'efficientnet_b0':
            self.backbone = models.efficientnet_b0(pretrained=pretrained)
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(self.backbone.classifier[1].in_features, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, num_classes)
            )

    def forward(self, x):
        return torch.sigmoid(self.backbone(x))

# Load model
@st.cache_resource
def load_model(model_path=None):
    model = CrackDetectionModel('mobilenet_v2', pretrained=False, num_classes=1)
    
    if model_path and os.path.exists(model_path):
        try:
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)
            st.success("✅ Pre-trained model loaded successfully!")
        except Exception as e:
            st.warning(f"⚠️ Could not load pre-trained weights: {e}")
            st.info("Using model with random weights. Please upload a trained model file.")
    else:
        st.warning("⚠️ No model file found. Using model with random weights.")
        st.info("Please upload a trained model file for better results.")
    
    model.to(device)
    model.eval()
    return model

# Image preprocessing
def get_transforms():
    return A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

# Prediction function
def predict_crack(image, model, transform, confidence_threshold=0.5):
    """Predict if image contains cracks"""
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    if len(image.shape) == 3 and image.shape[2] == 3:
        image_rgb = image
    else:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Apply transforms
    transformed = transform(image=image_rgb)
    input_tensor = transformed['image'].unsqueeze(0).to(device)
    
    # Make prediction
    with torch.no_grad():
        output = model(input_tensor)
        confidence = output.item()
        prediction = "Crack Detected" if confidence > confidence_threshold else "No Crack"
    
    return prediction, confidence

# Visualization function
def create_confidence_chart(confidence, prediction):
    """Create a confidence visualization"""
    color = "red" if prediction == "Crack Detected" else "green"
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = confidence * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Crack Confidence (%)"},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 100], 'color': "gray"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    
    fig.update_layout(height=300)
    return fig

def main():
    st.title("🔍 Crack Detection System")
    st.markdown("Detect cracks in concrete surfaces using deep learning")
    
    # Sidebar
    st.sidebar.header("⚙️ Settings")
    
    # Model upload
    uploaded_model = st.sidebar.file_uploader(
        "Upload trained model (.pth)",
        type=['pth'],
        help="Upload your trained crack detection model"
    )
    
    model_path = None
    if uploaded_model is not None:
        # Save uploaded model temporarily
        model_path = "temp_model.pth"
        with open(model_path, "wb") as f:
            f.write(uploaded_model.read())
    
    # Load model
    model = load_model(model_path)
    transform = get_transforms()
    
    # Confidence threshold
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.5,
        step=0.05,
        help="Threshold for crack detection decision"
    )
    
    # Mode selection
    mode = st.sidebar.selectbox(
        "Select Mode",
        ["📁 Upload Images", "📸 Live Camera"],
        help="Choose between image upload or live camera detection"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info(f"🖥️ Using: {device}")
    
    # Main content
    if mode == "📁 Upload Images":
        st.header("Upload Images for Crack Detection")
        
        uploaded_files = st.file_uploader(
            "Choose images...",
            type=['png', 'jpg', 'jpeg'],
            accept_multiple_files=True,
            help="Upload one or more images to detect cracks"
        )
        
        if uploaded_files:
            # Create columns for layout
            cols = st.columns(min(len(uploaded_files), 3))
            
            results = []
            for idx, uploaded_file in enumerate(uploaded_files):
                col = cols[idx % 3]
                
                with col:
                    # Load and display image
                    image = Image.open(uploaded_file)
                    st.image(image, caption=f"📷 {uploaded_file.name}", use_column_width=True)
                    
                    # Make prediction
                    with st.spinner("Analyzing..."):
                        prediction, confidence = predict_crack(
                            image, model, transform, confidence_threshold
                        )
                    
                    # Display results
                    if prediction == "Crack Detected":
                        st.error(f"🚨 {prediction}")
                    else:
                        st.success(f"✅ {prediction}")
                    
                    st.write(f"**Confidence:** {confidence:.2%}")
                    
                    # Store results
                    results.append({
                        'filename': uploaded_file.name,
                        'prediction': prediction,
                        'confidence': confidence
                    })
            
            # Summary
            if results:
                st.markdown("---")
                st.subheader("📊 Analysis Summary")
                
                col1, col2, col3 = st.columns(3)
                
                crack_count = sum(1 for r in results if r['prediction'] == "Crack Detected")
                avg_confidence = np.mean([r['confidence'] for r in results])
                
                with col1:
                    st.metric("Total Images", len(results))
                with col2:
                    st.metric("Cracks Detected", crack_count)
                with col3:
                    st.metric("Avg Confidence", f"{avg_confidence:.2%}")
                
                # Detailed results table
                with st.expander("📋 Detailed Results"):
                    import pandas as pd
                    df = pd.DataFrame(results)
                    df['confidence'] = df['confidence'].apply(lambda x: f"{x:.2%}")
                    st.dataframe(df, use_container_width=True)
    
    elif mode == "📸 Live Camera":
        st.header("Live Camera Crack Detection")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            camera_input = st.camera_input("Take a picture")
        
        with col2:
            st.markdown("### 📋 Instructions")
            st.markdown("""
            1. Click the camera button
            2. Take a photo of the surface
            3. Wait for analysis results
            4. Check confidence score
            """)
        
        if camera_input is not None:
            # Load and display the captured image
            image = Image.open(camera_input)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(image, caption="📸 Captured Image", use_column_width=True)
            
            with col2:
                # Make prediction
                with st.spinner("🔄 Analyzing image..."):
                    prediction, confidence = predict_crack(
                        image, model, transform, confidence_threshold
                    )
                
                # Display results
                st.markdown("### 🎯 Analysis Results")
                
                if prediction == "Crack Detected":
                    st.error(f"🚨 **{prediction}**")
                    st.markdown(
                        f"<div class='metric-container'>"
                        f"<h3>⚠️ CRACK ALERT</h3>"
                        f"<p>Confidence: {confidence:.2%}</p>"
                        f"</div>", 
                        unsafe_allow_html=True
                    )
                else:
                    st.success(f"✅ **{prediction}**")
                    st.markdown(
                        f"<div style='background: linear-gradient(90deg, #11998e 0%, #38ef7d 100%); padding: 1rem; border-radius: 10px; color: white;'>"
                        f"<h3>✅ NO CRACKS</h3>"
                        f"<p>Confidence: {confidence:.2%}</p>"
                        f"</div>", 
                        unsafe_allow_html=True
                    )
                
                # Confidence gauge
                fig = create_confidence_chart(confidence, prediction)
                st.plotly_chart(fig, use_container_width=True)
                
                # Additional info
                st.markdown("### 📈 Analysis Details")
                st.write(f"**Model Confidence:** {confidence:.4f}")
                st.write(f"**Threshold Used:** {confidence_threshold}")
                st.write(f"**Decision:** {'Above' if confidence > confidence_threshold else 'Below'} threshold")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "🏗️ Crack Detection System | Built with Streamlit & PyTorch"
        "</div>", 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    import os
    main()