import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import io
import os
import sys
import tempfile
from matplotlib import colors
from IrisRecognition import IrisSegmentation

def process_image(image_path, x_i, x_p):
    """Process the image with the given X_I and X_P parameters"""
    segmentation = IrisSegmentation(image_path=image_path)
    segmentation.to_grayscale()
    
    iris_threshold, pupil_threshold = segmentation.compute_threshold(X_I=x_i, X_P=x_p)
    segmentation.binarize_pupil(pupil_threshold)
    segmentation.binarize_iris(iris_threshold)
    
    segmentation.detect_pupil()
    segmentation.detect_iris()
    
    visualization = segmentation.visualize_segmentation()
    
    return visualization, segmentation

def main():
    st.set_page_config(layout="wide")
    
    st.title("Iris Segmentation App")
    
    with st.sidebar:
        st.header("Upload Image")
        uploaded_file = st.file_uploader("Choose an eye image", type=["jpg", "jpeg", "png"])
        
        st.header("Segmentation Parameters")
        x_i = st.slider("X_I (Iris Threshold Factor)", 1.5, 4.0, 2.2, 0.1, key="x_i_slider")
        x_p = st.slider("X_P (Pupil Threshold Factor)", 2.5, 8.0, 4.5, 0.1, key="x_p_slider")
    
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            temp_path = tmp_file.name
        
        original_image = Image.open(uploaded_file)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            st.image(original_image, use_container_width=True)
        
        try:
            with st.spinner("Processing..."):
                segmented_image, segmentation = process_image(temp_path, x_i, x_p)
                
                with col2:
                    st.subheader("Segmented Image")
                    st.image(segmented_image, use_container_width=True)
            
            col_info1, col_info2, col_info3 = st.columns(3)
            with col_info1:
                st.metric("Pupil Center", f"{segmentation.pupil_center}")
            with col_info2:
                st.metric("Pupil Radius", f"{segmentation.pupil_radius}")
            with col_info3:
                st.metric("Iris Radius", f"{segmentation.iris_radius}")
            
            tab1, tab2 = st.tabs(["Unwrapped Iris", "Iris Code"])
            
            with tab1:
                unwrapped = segmentation.unwrap_iris()
                st.image(unwrapped, caption="Unwrapped Iris", use_container_width=True)
            
            with tab2:
                with st.spinner("Generating iris code..."):
                    iris_code = segmentation.generate_iris_code()
                    
                    fig, ax = plt.subplots(figsize=(12, 4))
                    ax.imshow(iris_code, cmap='gray', interpolation='nearest', aspect='auto')
                    ax.set_title("Iris Code Map")
                    ax.set_xlabel("Angular bits (real + imag)")
                    ax.set_ylabel("Radial bands")
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.grid(False)
                    plt.tight_layout()
                    st.pyplot(fig)
                    
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
            st.info("Try adjusting the threshold parameters.")
        
        try:
            os.unlink(temp_path)
        except:
            pass
    else:
        st.info("Please upload an eye image to begin.")
        
        st.markdown("""
        ### How to use this app:
        1. Upload an eye image using the sidebar
        2. Adjust the threshold parameters:
           - **X_I**: Controls the iris threshold (lower value = more area detected)
           - **X_P**: Controls the pupil threshold (higher value = smaller pupil)
        3. The segmentation updates automatically
        """)

if __name__ == "__main__":
    main()