import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.impute import SimpleImputer

# Import all models and metrics
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report, ConfusionMatrixDisplay

# For Unsupervised Clustering
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Imports for Dendrogram
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

# Imports for Tabular CNN
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, Dropout, MaxPooling1D

# ---------------------------
# BEST PARAMETERS (FOR DEMO)
# ---------------------------

BEST_PARAMS_FOR_DEMO = {
    'LogisticRegression': {
        'solver': 'liblinear', 
        'penalty': 'l2', 
        'C': 10
    },
    'KNN': {
        'weights': 'distance', 
        'n_neighbors': 13, 
        'metric': 'euclidean'
    },
    'SVM': {
        'kernel': 'rbf', 
        'gamma': 0.001, 
        'C': 100
    },
    'DecisionTree': {
        'min_samples_split': 15, 
        'min_samples_leaf': 1, 
        'max_features': 'log2', 
        'max_depth': 20, 
        'criterion': 'gini'
    },
    'RandomForest': {
        'n_estimators': 1000, 
        'min_samples_split': 2, 
        'min_samples_leaf': 1, 
        'max_features': 'log2', 
        'max_depth': 10, 
        'criterion': 'entropy', 
        'bootstrap': True
    }
}

# ---------------------------
# DATA PROCESSING FUNCTIONS
# ---------------------------

@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    return df

@st.cache_data
def preprocess_data(df):
    """
    Preprocesses the data, scales it, and also returns the unscaled
    data for the CNN comparison.
    """
    if 'id' in df.columns:
        df = df.drop(columns=['id'])
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    target_col = 'diagnosis'
    le = LabelEncoder()
    y_original_labels = df[target_col].copy() 
    df[target_col] = le.fit_transform(df[target_col])
    
    X = df.drop(columns=[target_col])
    y = df[target_col].values
    
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    
    
    # Handle missing values before outlier detection
    imputer = SimpleImputer(strategy='median')
    X[num_cols] = imputer.fit_transform(X[num_cols])
    outlier_df = outlier_report_iqr(X, cols=num_cols)
    # --- IMPORTANT ---
    # Get unscaled data (as numpy array) AFTER imputation
    X_unscaled = X.values 
    
    # Create the scaler and fit it
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X) 
    
    # Split scaled data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y)
    
    # Split unscaled data using the SAME random state
    X_train_unscaled, X_test_unscaled, _, _ = train_test_split(
        X_unscaled, y, test_size=0.2, random_state=42, stratify=y)

    # Return all 13 values
    return (
        X_train, X_test, y_train, y_test, 
        X_train_unscaled, X_test_unscaled, 
        X_scaled, y, num_cols, 
        outlier_df, le.classes_, y_original_labels, scaler
    )

def outlier_report_iqr(X_df, cols=None, factor=1.5):
    if cols is None:
        cols = X_df.columns
    summary = []
    # (Assuming outlier_report_iqr is defined here as in your file...)
    # ... (omitting full code for brevity, your version is fine)
    # This is just a placeholder, use your working function
    report = pd.DataFrame(summary, columns=["Column", "Outliers", "% Outliers", "Lower Bound", "Upper Bound", "Min", "Max"])
    return report

# ---------------------------
# ML TRAINING FUNCTIONS
# ---------------------------

@st.cache_resource
def train_models_for_demo(X_train, y_train):
    """
    Trains all models using pre-defined, optimized hyperparameters
    for a fast demonstration. Skips all searching.
    """
    print("--- Running FAST demonstration training ---")
    
    # 1. Get the pre-set, best parameters
    params_lr = BEST_PARAMS_FOR_DEMO.get('LogisticRegression', {})
    params_knn = BEST_PARAMS_FOR_DEMO.get('KNN', {})
    params_svm = BEST_PARAMS_FOR_DEMO.get('SVM', {})
    params_dt = BEST_PARAMS_FOR_DEMO.get('DecisionTree', {})
    params_rf = BEST_PARAMS_FOR_DEMO.get('RandomForest', {})

    # 2. Define the models with their BEST parameters
    models = {}
    models['LogisticRegression'] = LogisticRegression(
        max_iter=500, class_weight='balanced', random_state=42, **params_lr
    )
    models['KNN'] = KNeighborsClassifier(**params_knn)
    models['SVM'] = SVC(
        probability=True, random_state=42, **params_svm
    )
    models['DecisionTree'] = DecisionTreeClassifier(
        random_state=42, class_weight='balanced', **params_dt
    )
    models['RandomForest'] = RandomForestClassifier(
        random_state=42, class_weight='balanced', **params_rf
    )

    # 3. Just call .fit() on every model. No search!
    best_models = {}
    for name, model in models.items():
        print(f"Fitting {name} with best params...")
        model.fit(X_train, y_train)
        best_models[name] = model
        
    print("--- Model fitting complete ---")
    return best_models

@st.cache_resource
def train_models(X_train, y_train):
    """
    (This is the SLOW research function)
    Trains multiple models using RandomizedSearchCV with expanded parameter grids
    to find the best hyperparameters efficiently.
    """
    
    # 1. Define the models
    models = {}
    models['LogisticRegression'] = LogisticRegression(max_iter=500, class_weight='balanced', random_state=42, solver='liblinear')
    models['KNN'] = KNeighborsClassifier()
    models['SVM'] = SVC(probability=True, random_state=42)
    models['DecisionTree'] = DecisionTreeClassifier(random_state=42, class_weight='balanced')
    models['RandomForest'] = RandomForestClassifier(random_state=42, class_weight='balanced')

    # 2. Define the expanded parameter grids
    param_grids_expanded = {
        # ... (Your deep grids are here) ...
    'KNN': {
        'n_neighbors': [3, 5, 7, 9, 11, 13, 15, 17, 19, 21],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'minkowski']
    },
    'SVM': {
        'C': [0.1, 1, 10, 100, 500, 1000],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1] 
    },
    'LogisticRegression': {
        'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear']
    },
    'DecisionTree': {
        'criterion': ['gini', 'entropy'],
        'max_depth': [5, 10, 15, 20, 30, 50, None],
        'min_samples_split': [2, 5, 10, 15, 20],
        'min_samples_leaf': [1, 2, 4, 6, 8, 10],
        'max_features': ['sqrt', 'log2', None]
    },
    'RandomForest': {
        'n_estimators': [100, 200, 300, 500, 800, 1000],
        'criterion': ['gini', 'entropy'],
        'max_depth': [5, 10, 15, 20, 30, 50, None],
        'min_samples_split': [2, 5, 10, 15],
        'min_samples_leaf': [1, 2, 4, 6, 8],
        'max_features': ['sqrt', 'log2', None],
        'bootstrap': [True, False]
    }
}
    
    # 3. Loop, search, and train
    best_models = {}
    for name, model in models.items():
        if name in param_grids_expanded:
            rand_search = RandomizedSearchCV(
                model, 
                param_grids_expanded[name], 
                cv=5, 
                scoring='roc_auc', 
                n_jobs=-1,
                n_iter=50, # Using 50 as you set
                random_state=42
            )
            print(f"Running RandomizedSearch for {name}...")
            rand_search.fit(X_train, y_train)
            best_models[name] = rand_search.best_estimator_
            print(f"Best params for {name}: {rand_search.best_params_}")
            
        else:
            print(f"Running standard .fit() for {name}...")
            model.fit(X_train, y_train)
            best_models[name] = model
            
    print("Model training complete.")
    return best_models

# ---------------------------
# XAI & CLUSTERING FUNCTIONS
# ---------------------------

@st.cache_data
def get_feature_importances(_models, feature_names):
    importances = {}
    for name, model in _models.items():
        if hasattr(model, 'feature_importances_'):
            imp = model.feature_importances_
        elif hasattr(model, 'coef_'):
            imp = model.coef_[0]
        else:
            continue
        
        imp_df = pd.DataFrame({'feature': feature_names, 'importance': np.abs(imp)})
        imp_df = imp_df.sort_values('importance', ascending=False).head(10)
        importances[name] = imp_df
        
    return importances

@st.cache_data
def run_pca(_scaled_data, n_components=2):
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(_scaled_data)
    pca_df = pd.DataFrame(data=pca_result, columns=[f'PC {i+1}' for i in range(n_components)])
    return pca_df, pca.explained_variance_ratio_

@st.cache_data
def run_kmeans(_scaled_data, k=2):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(_scaled_data)
    return clusters

@st.cache_data
def plot_feature_dendrogram(_scaled_data, feature_names):
    """
    Performs hierarchical clustering on the *features* (not samples)
    and returns a matplotlib figure of the dendrogram.
    """
    linkage_matrix = linkage(_scaled_data.T, method='ward')
    fig, ax = plt.subplots(figsize=(15, 10))
    dendrogram(
        linkage_matrix,
        labels=feature_names,
        leaf_rotation=90,
        leaf_font_size=10,
        ax=ax
    )
    ax.set_title('Hierarchical Clustering Dendrogram of Features', fontsize=16)
    ax.set_ylabel('Cluster Distance (Ward)', fontsize=12)
    plt.tight_layout()
    
    return fig

# ---------------------------
# TABULAR CNN FUNCTIONS
# ---------------------------

@st.cache_resource
def train_cnn_tabular_adam(X_train, y_train):
    """
    Trains a 1D CNN on SCALED data using the 'Adam' optimizer.
    """
    X_train_cnn = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    input_shape = (X_train.shape[1], 1)
    
    model = Sequential([
        Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(50, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train_cnn, y_train, epochs=30, batch_size=32, verbose=0, validation_split=0.2)
    return model, history

@st.cache_resource
def train_cnn_tabular_sgd(X_train, y_train):
    """
    Trains the SAME 1D CNN on UNSCALED data using the 'SGD' optimizer.
    """
    X_train_cnn = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    input_shape = (X_train.shape[1], 1)
    
    model = Sequential([
        Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(50, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train_cnn, y_train, epochs=30, batch_size=32, verbose=0, validation_split=0.2)
    return model, history