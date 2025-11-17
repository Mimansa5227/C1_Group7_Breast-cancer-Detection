import tensorflow as tf
import os
import streamlit as st
import cv2
import numpy as np

from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
from tf_keras_vis.utils.scores import BinaryScore
import matplotlib.cm

CNN_MODEL_PATH = os.path.join('models', 'breast_cance_model.keras')

@st.cache_resource
def load_cnn_model():
    """
    Loads the complete saved CNN model from the 'models' folder.
    """
    
    if not os.path.exists(CNN_MODEL_PATH):
        st.error(f"FATAL: Model file not found at {CNN_MODEL_PATH}")
        st.error("Please make sure the 'models' folder exists and contains 'breast_cance_model.keras'.")
        return None 
    
    try:
        
        model = tf.keras.models.load_model(CNN_MODEL_PATH)
        
        print("Complete model (architecture and weights) loaded successfully.")
        return model
        
    except Exception as e:
        st.error(f"Error loading complete CNN model: {e}")
        return None
    
def generate_gradcam(model, img_batch, img_resized, probability):
    """
    Generates a Grad-CAM heatmap and overlays it on the original image.
    """
    try:
        
        last_conv_layer = 'conv5_block3_out' 

        score_value = 1 if probability >= 0.5 else 0
        score = BinaryScore(score_value)

        cam = Gradcam(model, model_modifier=ReplaceToLinear(), clone=True)

      
        heatmap = cam(score,
                      img_batch, # The (1, 224, 224, 3) float32 tensor
                      penultimate_layer=last_conv_layer)
        
        heatmap = heatmap[0] 
        jet_heatmap = np.uint8(matplotlib.cm.jet(heatmap)[..., :3] * 255)
        
        jet_heatmap_resized = cv2.resize(jet_heatmap, (img_resized.shape[1], img_resized.shape[0]))

        overlay = cv2.addWeighted(img_resized, 0.6, jet_heatmap_resized, 0.4, 0)
        
        return overlay

    except Exception as e:
        st.error(f"Grad-CAM Error: {e}")
        # This can happen if the layer name 'conv5_block3_out' is wrong
        st.warning("Could not find 'conv5_block3_out'. Is this a ResNet50 model?")
        return None
    
def cnn_predict_image(uploaded_file, trained_model):
    """
    Processes the uploaded image and makes a prediction.
    (This is your fixed function)
    """
    
    # 1. Decode the image file
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is None:
        st.error("Error: Could not decode the image file.")
        return

    # 2. Preprocess the Image
    img_resized = cv2.resize(img, (224, 224))
    
    if len(img_resized.shape) == 2 or img_resized.shape[2] == 1:
        img_resized = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2BGR)

    # --- MATCHING COLAB PREPROCESSING ---
    # Convert to float, but NO division by 255.0
    img_processed = img_resized.astype(np.float32)
    
    # Reshape for the model: (1, 224, 224, 3)
    img_batch = np.expand_dims(img_processed, axis=0)

    # 3. Make the Prediction
    prediction = trained_model.predict(img_batch, verbose=0)
    probability = prediction[0][0] # Probability of CLASS 1

    # 4. Display the Result
    st.subheader("CNN Image Prediction Result")
    
    col_img, col_pred = st.columns([1, 2])
    with col_img:
        st.image(img_resized, caption="Processed Image (224x224)", use_column_width=True)

    with col_pred:
        st.markdown(f"**Filename:** `{uploaded_file.name}`")
        
        label_name = "" # To store the label for the heatmap caption
        
        # --- PREDICTION LOGIC (YOURS IS CORRECT) ---
        if probability >= 0.5:
            label_name = "MALIGNANT"
            st.markdown(f"### <span style='color:red;'>🚨 Prediction: {label_name}</span>", unsafe_allow_html=True)
            st.metric(label=f"Confidence (Model predicts {label_name})", value=f"{100 * probability:.2f}%")
        else:
            label_name = "BENIGN"
            st.markdown(f"### <span style='color:green;'>✅ Prediction: {label_name}</span>", unsafe_allow_html=True)
            st.metric(label=f"Confidence (Model predicts {label_name})", value=f"{100 * (1 - probability):.2f}%")
        
        st.markdown(f"<small>This prediction uses the pre-trained CNN model (`{CNN_MODEL_PATH}`).</small>", unsafe_allow_html=True)

        # --- NEW GRAD-CAM SECTION ---
        st.markdown("---")
        st.subheader(" Model Reasoning (Grad-CAM)")
        
        # We add a spinner while it calculates the heatmap
        with st.spinner("Generating heatmap..."):
            
            # Call our new helper function
            heatmap_overlay = generate_gradcam(
                trained_model,
                img_batch,       # The (1, 224, 224, 3) float32 tensor
                img_resized,     # The (224, 224, 3) uint8 image
                probability
            )
            
            if heatmap_overlay is not None:
                st.image(heatmap_overlay, caption=f"Heatmap: Why the model predicted '{label_name}'")
                st.info("The **brighter red** areas are what the model focused on most to make its decision.")