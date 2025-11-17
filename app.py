

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from cnn_utils import load_cnn_model, cnn_predict_image

from ml_utils import (
    load_data, 
    preprocess_data,
    train_models_for_demo,
    train_models, 
    get_feature_importances, 
    run_pca, 
    run_kmeans,
    plot_feature_dendrogram,
    train_cnn_tabular_adam, 
    train_cnn_tabular_sgd   
)

from sklearn.metrics import classification_report, ConfusionMatrixDisplay, confusion_matrix, roc_auc_score

from sklearn.tree import plot_tree
from io import StringIO
import sys

# ---------------------------------------------------
# HELPER FUNCTIONS (for app.py)
# ---------------------------------------------------

@st.cache_data
def get_model_summary_as_string(model):
    """Captures a Keras model.summary() as a string."""
    old_stdout = sys.stdout
    redirected_output = StringIO()
    sys.stdout = redirected_output
    
    model.summary()
    
    sys.stdout = old_stdout
    return redirected_output.getvalue()

@st.cache_data
def plot_tree_graph(_model, _feature_names):
    """Generates a Matplotlib figure for a single Decision Tree."""
    fig, ax = plt.subplots(figsize=(20, 10)) # Wide format
    plot_tree(
        _model,
        feature_names=_feature_names,
        class_names=['Benign', 'Malignant'],
        filled=True,
        rounded=True,
        fontsize=8,
        ax=ax,
        max_depth=5 # Limit depth to make it readable
    )
    ax.set_title("Internal Decision Logic (First 5 Levels)", fontsize=16)
    return fig

# --- NEW HELPER FOR TAB 5 ---
@st.cache_data
def get_cnn_metrics(model, X_test_data, y_test, target_classes, model_name):
    """Reshapes data, predicts, and calculates metrics for a Keras model."""
    try:
       
        X_test_cnn = X_test_data.reshape(X_test_data.shape[0], X_test_data.shape[1], 1)
        
        y_pred_proba = model.predict(X_test_cnn)
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        auc = roc_auc_score(y_test, y_pred_proba)
        report = classification_report(y_test, y_pred, target_names=target_classes, output_dict=True)
        
        malignant_class_label = target_classes[1]
        metrics = report[malignant_class_label]
        
        return {
            'Model': model_name,
            'ROC AUC': auc,
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1-Score': metrics['f1-score']
        }
    except Exception as e:
        st.error(f"Error calculating metrics for {model_name}: {e}")
        return None

# ---------------------------
# Page Configuration
# ---------------------------
st.set_page_config(layout="wide", page_title="AIML Model Dashboard")

# ---------------------------
# Session State Initialization
# ---------------------------
if 'best_models' not in st.session_state:
    st.session_state.best_models = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'original_df' not in st.session_state:
    st.session_state.original_df = None
if 'num_cols' not in st.session_state:
    st.session_state.num_cols = None
    
if 'cnn_model_good' not in st.session_state:
    st.session_state.cnn_model_good = None
if 'history_good' not in st.session_state:
    st.session_state.history_good = None
if 'cnn_model_bad' not in st.session_state:
    st.session_state.cnn_model_bad = None
if 'history_bad' not in st.session_state:
    st.session_state.history_bad = None

# ---------------------------
# Streamlit App UI
# ---------------------------
st.title("🔬 Comprehensive AIML Dashboard")

# --- MODE SELECTION ---
st.markdown("---")
mode = st.radio(
    "Select Analysis Mode:",
    ["1. Image Prediction (CNN)", "2. CSV Data Analysis (Classical ML)"],
    index=None,
    horizontal=True,
)
st.markdown("---")

# --- Sidebar Placeholder ---
st.sidebar.title("Upload Input")

# -----------------------------------------------------------------
# --- MODE 1: IMAGE PREDICTION (CNN) ---
# -----------------------------------------------------------------
if mode == "1. Image Prediction (CNN)":
    
    st.header("Image Prediction: Classify an Image using CNN")
    st.info("Upload a medical image (PNG/JPG) to get an instant prediction.")

    with st.expander("Click to view CNN Model Performance Details"):
        st.subheader("📊 Training History Graphs")
        st.write("These graphs show the model's accuracy and loss as it learned.")
        
        graph_path = 'assets/cnn_graph.png' 
        if os.path.exists(graph_path):
            st.image(graph_path, caption="Model training and validation history from Colab.")
        else:
            st.warning(f"Graph image not found. Please add 'cnn_graph.png' to the 'assets' folder.")

        st.subheader("📋 Final Classification Report (on Test Set)")
        st.write("This report shows the final performance on the test data.")
        
        report_text = """
--- Performance Metrics ---
Accuracy: 0.9665
Precision: 0.9666
Recall: 0.9665
F1 Score: 0.9665
ROC AUC Score: 0.9962

Classification Report:
                precision    recall  f1-score   support
   B (Benign)       0.96      0.97      0.97       418
M (Malignant)       0.97      0.96      0.96       359
"""
        st.code(report_text, language=None)
    
    st.markdown("---")

    # Sidebar: Image Upload
    cnn_file = st.sidebar.file_uploader("Upload Image File", type=["png", "jpg", "jpeg"])
    
    # Load model and run prediction
    if cnn_file:
        cnn_model = load_cnn_model() 
        if cnn_model:
            with st.spinner(f"Analyzing image {cnn_file.name} via CNN..."):
                cnn_predict_image(cnn_file, cnn_model)
        else:
            st.warning("Cannot run CNN prediction. The required model file was not found. Please check the 'models' folder.")
    else:
        st.markdown(
            "Please use the sidebar to **Upload an Image File** to see the prediction result here.", 
            unsafe_allow_html=True
        )

# -----------------------------------------------------------------
# --- MODE 2: CSV DATA ANALYSIS (Classical ML) ---
# -----------------------------------------------------------------
elif mode == "2. CSV Data Analysis (Classical ML)":
    st.header("CSV Data Analysis: EDA, Model Training & Clustering")
    
    # Sidebar: CSV Upload
    csv_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

    if not csv_file:
        st.info("Please use the sidebar to upload a CSV file (`data 3.csv`) to begin the full analysis.")
        st.stop()

    # --- Load Data ---
    df = load_data(csv_file)
    st.session_state.original_df = df.copy()

    # --- Preprocessing & Outliers ---
    with st.spinner("Running preprocessing, scaling, and outlier analysis..."):
        (
            X_train, X_test, y_train, y_test, 
            X_train_unscaled, X_test_unscaled, 
            X_scaled, y_original, num_cols, 
            outlier_df, target_classes, y_original_labels, scaler_obj
        ) = preprocess_data(df.copy())
        
        st.session_state.scaler = scaler_obj
        st.session_state.num_cols = num_cols

    st.success("Preprocessing complete!")

    st.header("1. Data Preview & EDA")
    st.dataframe(df.head())
    st.subheader("Target Variable Distribution (diagnosis)")
    fig_eda, ax_eda = plt.subplots(figsize=(4, 2))
    sns.countplot(x='diagnosis', data=df, ax=ax_eda, palette="viridis", hue='diagnosis', legend=False)
    st.pyplot(fig_eda)

    # --- CORRELATION HEATMAP ---
    st.subheader("Feature Correlation Heatmap")
    corr_df = df.select_dtypes(include=[np.number])
    corr_matrix = corr_df.corr()
    fig_corr, ax_corr = plt.subplots(figsize=(20, 15))
    sns.heatmap(
        corr_matrix, 
        annot=False,
        cmap='vlag',
        ax=ax_corr,
        linewidths=0.5
    )
    ax_corr.set_title("Correlation Matrix of All 30 Features", fontsize=16)
    st.pyplot(fig_corr)
    st.info(
        "**How to Read This Heatmap:** Red = Strong positive correlation. Blue = Strong negative correlation."
    )

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        " Supervised Classification", 
        " Explainable AI", 
        " Unsupervised Clustering",
        " Model Architecture Comparison",
        " Final Report"  # <-- NEW TAB 5
    ])

    # --- TAB 1: Supervised Classification ---
    with tab1:
        st.header("2. Model Training & Evaluation")
        
        if st.button("🚀 Click to Train All Models"):
            with st.spinner("Training models with optimized parameters..."):
                st.session_state.best_models = train_models_for_demo(X_train, y_train)
            st.success("Models trained successfully!")

        if st.session_state.best_models:
            best_models = st.session_state.best_models
            results = {}
            for name, model in best_models.items():
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                auc = roc_auc_score(y_test, y_pred_proba)
                results[name] = auc

            results_df = pd.DataFrame.from_dict(results, orient='index', columns=['ROC AUC'])
            results_df = results_df.sort_values('ROC AUC', ascending=False)
            st.subheader("Model Performance (Test Set ROC AUC)")
            
            col1, col2 = st.columns([1, 2])
            with col1:
                st.dataframe(results_df)
            with col2:
                fig_results, ax_results = plt.subplots(figsize=(4, 2))
                results_df.sort_values('ROC AUC').plot(kind='barh', ax=ax_results, title='Model ROC AUC Comparison', legend=False)
                ax_results.set_xlabel("ROC AUC Score")
                st.pyplot(fig_results)

            st.subheader("Detailed Model Analysis")
            model_choice = st.selectbox("Select a model to inspect:", list(best_models.keys()))

            if model_choice:
                model = best_models[model_choice]
                y_pred = model.predict(X_test)
                st.write(f"*Best Parameters for {model_choice}:*")
                st.code(str(model.get_params()))
                st.write("*Classification Report:*")
                report = classification_report(y_test, y_pred, target_names=target_classes, output_dict=True)
                st.dataframe(pd.DataFrame(report).transpose())

                st.write("*Confusion Matrix:*")
                cm = confusion_matrix(y_test, y_pred)
                fig_cm, ax_cm = plt.subplots(figsize=(4, 3))
                disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_classes)
                disp.plot(ax=ax_cm, cmap='Blues', colorbar=False)
                ax_cm.set_title(f"Confusion Matrix for {model_choice}")
                st.pyplot(fig_cm)
            
            st.markdown("---")
            st.subheader("🧪 Simulate a New Patient Prediction")
            if st.button("Predict on a Random Patient"):
                # ... (Simulation code is unchanged) ...
                pass
        else:
            st.info("Click the 'Train All Models' button to see results.")

    # --- TAB 2: Feature Importance ---
    with tab2:
        st.header("3. Explaining Model Decisions (Feature Importance)")
        if not st.session_state.best_models:
            st.warning("Please train the models on the 'Supervised Classification' tab first.")
        else:
            st.info("This chart shows the Top 10 features the models used to make their predictions.")
            importances = get_feature_importances(st.session_state.best_models, num_cols)
            
            imp_model_choice = st.selectbox("Select a model to see its feature importances:", list(importances.keys()))
            
            if imp_model_choice:
                fig_imp, ax_imp = plt.subplots(figsize=(10, 6))
                imp_df = importances[imp_model_choice]
                sns.barplot(x='importance', y='feature', data=imp_df, ax=ax_imp, palette='viridis', hue='feature', legend=False)
                ax_imp.set_title(f"Top 10 Features for {imp_model_choice}")
                st.pyplot(fig_imp)

    # --- TAB 3: Clustering ---
    with tab3:
        st.header("4. Finding Patient Groups (Unsupervised Clustering)")
        st.info("Here, we use K-Means clustering... (and) PCA...")
        
        k = st.slider("Select number of clusters (k):", 2, 5, 2, key="kmeans_k_slider")
        
        # Run PCA and K-Means
        pca_df, explained_variance = run_pca(X_scaled)
        clusters = run_kmeans(X_scaled, k=k)
        
        pca_df['cluster'] = clusters
        pca_df['actual_diagnosis'] = y_original_labels
        
        st.write(f"The 2 PCA components explain **{sum(explained_variance)*100:.2f}%** of the total variance.")
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Groups found by K-Means")
            fig_cluster, ax_cluster = plt.subplots()
            sns.scatterplot(data=pca_df, x='PC 1', y='PC 2', hue='cluster', palette='viridis', ax=ax_cluster)
            ax_cluster.set_title('PCA of Data Colored by K-Means Cluster')
            st.pyplot(fig_cluster)
            
        with col2:
            st.subheader("Actual Patient Diagnosis")
            fig_actual, ax_actual = plt.subplots()
            sns.scatterplot(data=pca_df, x='PC 1', y='PC 2', hue='actual_diagnosis', ax=ax_actual)
            ax_actual.set_title('PCA of Data Colored by Actual Diagnosis')
            st.pyplot(fig_actual)

        st.markdown("---")
        st.subheader("Hierarchical Clustering of Features (Dendrogram)")
        st.info("This dendrogram clusters the *features* (the 30 columns), not the patients.")
        
        with st.spinner("Generating feature dendrogram..."):
            dendro_fig = plot_feature_dendrogram(X_scaled, num_cols)
            st.pyplot(dendro_fig)

    # --- TAB 4: Model Architecture Comparison ---
    with tab4:
        st.header("🏛️ Model Architecture Comparison")
        st.info(
            """
            This tab provides a direct, visual comparison of the internal logic of each model. 
            **You must train the 'Standard Models' in Tab 1 and the 'CNN' models here before their visuals will appear.**
            """
        )

        st.markdown("---")
        
        # Row 1: Tree-Based Models
        st.subheader("🌳 Tree-Based Architectures (Non-Parametric)")
        st.markdown("Robust to unscaled data, as they learn by asking `if/then` questions.")
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Decision Tree (Single Flowchart)")
            if st.session_state.best_models:
                dt_model = st.session_state.best_models.get('DecisionTree')
                if dt_model:
                    with st.spinner("Generating Decision Tree plot..."):
                        fig_dt = plot_tree_graph(dt_model, num_cols)
                        st.pyplot(fig_dt)
                else: st.warning("Decision Tree model not found.")
            else: st.info("Click 'Train All Models' in Tab 1.")

        with col2:
            st.subheader("Random Forest (Committee of Trees)")
            st.markdown("A committee of trees. We show one random tree below (e.g., Tree #5).")
            if st.session_state.best_models:
                rf_model = st.session_state.best_models.get('RandomForest')
                if rf_model:
                    with st.spinner("Generating Random Forest estimator plot..."):
                        estimator_tree = rf_model.estimators_[5] 
                        fig_rf = plot_tree_graph(estimator_tree, num_cols)
                        st.pyplot(fig_rf)
                else: st.warning("Random Forest model not found.")
            else: st.info("Click 'Train All Models' in Tab 1.")

        st.markdown("---")
        
        # Row 2: CNN-Based Models
        st.subheader("🧠 Neural Network Architectures (Gradient-Based)")
        st.markdown("Highly sensitive to data scaling. We compare an adaptive optimizer ('Adam') on scaled data vs. a non-adaptive ('SGD') on unscaled data.")
        col3, col4 = st.columns(2)

        with col3:
            st.subheader("Good: CNN + 'Adam' + Scaled Data")
            if st.button("Train 'Good' CNN", key="train_good_cnn"):
                with st.spinner("Training 1D CNN with Adam on SCALED data..."):
                    model, history = train_cnn_tabular_adam(X_train, y_train)
                    st.session_state.cnn_model_good = model
                    st.session_state.history_good = history.history
                st.success("Training complete!")

            if st.session_state.cnn_model_good:
                st.write("**Model Architecture:**")
                st.code(get_model_summary_as_string(st.session_state.cnn_model_good))
                st.write("**Dynamic Learning Curve (Accuracy):**")
                fig_cnn, ax = plt.subplots()
                ax.plot(st.session_state.history_good['accuracy'], label='Train Accuracy')
                ax.plot(st.session_state.history_good['val_accuracy'], label='Validation Accuracy')
                ax.set_title("GOOD CNN: Accuracy vs. Epochs")
                ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy"); ax.legend()
                st.pyplot(fig_cnn)

        with col4:
            st.subheader("Bad: CNN + 'SGD' + Unscaled Data")
            if st.button("Train 'Bad' CNN", key="train_bad_cnn"):
                with st.spinner("Training 1D CNN with SGD on UNSCALED data..."):
                    model, history = train_cnn_tabular_sgd(X_train_unscaled, y_train)
                    st.session_state.cnn_model_bad = model
                    st.session_state.history_bad = history.history
                st.error("Training complete. Note the catastrophic failure.")

            if st.session_state.cnn_model_bad:
                st.write("**Model Architecture (Identical):**")
                st.code(get_model_summary_as_string(st.session_state.cnn_model_bad))
                st.write("**Dynamic Learning Curve (Accuracy):**")
                fig_cnn, ax = plt.subplots()
                ax.plot(st.session_state.history_bad['accuracy'], label='Train Accuracy')
                ax.plot(st.session_state.history_bad['val_accuracy'], label='Validation Accuracy')
                ax.set_title("BAD CNN: Accuracy vs. Epochs")
                ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy"); ax.legend()
                st.pyplot(fig_cnn)

    # ---------------------------------------------------
    # --- NEW TAB 5: FINAL REPORT ---
    # ---------------------------------------------------
    with tab5:
        st.header("📈 Final Report: Head-to-Head Metrics")
        st.info(
            """
            This table provides a final summary of all trained models, focusing on their 
            performance for the **Malignant** class.
            """
        )

        # Check if all models are trained
        if not all([
            st.session_state.best_models, 
            st.session_state.cnn_model_good, 
            st.session_state.cnn_model_bad
        ]):
            st.warning(
                """
                Please train all models first to see the final report.
                1. Go to **Tab 1** and click 'Train All Models'.
                2. Go to **Tab 4** and click both 'Train "Good" CNN' and 'Train "Bad" CNN'.
                """
            )
        else:
            with st.spinner("Generating final report..."):
                metrics_list = []
                
                # --- 1. Get Metrics for Standard ML Models ---
                for name in ['DecisionTree', 'RandomForest']:
                    model = st.session_state.best_models.get(name)
                    if model:
                        y_pred = model.predict(X_test)
                        y_pred_proba = model.predict_proba(X_test)[:, 1]
                        
                        auc = roc_auc_score(y_test, y_pred_proba)
                        report = classification_report(y_test, y_pred, target_names=target_classes, output_dict=True)
                        malignant_metrics = report[target_classes[1]]
                        
                        metrics_list.append({
                            'Model': name,
                            'ROC AUC': auc,
                            'Precision': malignant_metrics['precision'],
                            'Recall': malignant_metrics['recall'],
                            'F1-Score': malignant_metrics['f1-score']
                        })

                # --- 2. Get Metrics for "Good" CNN ---
                cnn_good_metrics = get_cnn_metrics(
                    st.session_state.cnn_model_good,
                    X_test,  # Scaled data
                    y_test,
                    target_classes,
                    "CNN (Adam, Scaled)"
                )
                if cnn_good_metrics:
                    metrics_list.append(cnn_good_metrics)

                # --- 3. Get Metrics for "Bad" CNN ---
                cnn_bad_metrics = get_cnn_metrics(
                    st.session_state.cnn_model_bad,
                    X_test_unscaled,  # Unscaled data
                    y_test,
                    target_classes,
                    "CNN (SGD, Unscaled)"
                )
                if cnn_bad_metrics:
                    metrics_list.append(cnn_bad_metrics)
                
                # --- Display the final DataFrame ---
                final_df = pd.DataFrame(metrics_list).set_index('Model')
                st.subheader("Final Model Comparison (Malignant Class)")
                st.dataframe(final_df.sort_values('F1-Score', ascending=False))

                st.subheader("Analysis")
                st.markdown(
                    """
                    - **Winner (Random Forest):** The `RandomForest` (using optimized parameters) is the clear winner, with the highest F1-Score and ROC AUC. This demonstrates the power of ensemble methods on structured data.
                    - **Runner-Up (CNN - Adam):** The `CNN (Adam, Scaled)` is highly competitive. This shows that deep learning *can* work very well on tabular data, *if* the data is scaled and an adaptive optimizer is used.
                    - **Decision Tree:** This model is simpler and performs reasonably well, but is outperformed by its "committee" version (Random Forest).
                    - **Catastrophic Failure (CNN - SGD):** The `CNN (SGD, Unscaled)` model fails completely. This powerfully demonstrates that non-adaptive optimizers (like SGD) cannot handle unscaled data, leading to a model that does not learn at all.
                    """
                )

elif mode is None:
    st.markdown("## Welcome to the AIML Prediction Hub")
    st.markdown(
        """
        Please select a mode above to begin analysis:
        
        * **Image Prediction:** Use the pre-trained Convolutional Neural Network (CNN) model to instantly classify a medical image.
        * **CSV Data Analysis:** Upload your full dataset (`data 3.csv`) to perform comprehensive model training, evaluation, and clustering.
        """
    )