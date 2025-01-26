from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
import pandas as pd
import streamlit as st

def preprocess_data(data):
    """
    Splits the dataset into features (X) and target (y).
    
    Args:
        data (pd.DataFrame): The dataset with target column as the first column.
        
    Returns:
        tuple: Features (X) and target (y).
    """
    if data.empty:
        raise ValueError("The dataset is empty.")
    
    y = data.iloc[:, 0]  # Assuming the target is in the first column
    X = data.iloc[:, 1:]  # Features are all columns except the target
    return X, y

def run_ml_models(models, X, y):
    """
    Trains and evaluates multiple ML models using the provided data.
    
    Args:
        models (dict): A dictionary of pre-loaded ML models.
        X (pd.DataFrame): Features.
        y (pd.Series): Target labels.
        
    Returns:
        list: A list of dictionaries containing model evaluation metrics.
    """
    results = []
    # Check if there are enough samples
    if len(X) < 2:
        raise ValueError("Not enough samples in the dataset to split into training and testing sets.")
        
    # Split the data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    for model_name, model in models.items():
        st.write(f"Training {model_name}...")

        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            "name": model_name,
            "train_accuracy": accuracy_score(y_train, y_train_pred),
            "test_accuracy": accuracy_score(y_test, y_test_pred),
            "f1_score_train": f1_score(y_train, y_train_pred),
            "f1_score_test": f1_score(y_test, y_test_pred),
            "precision": precision_score(y_test, y_test_pred),
            "recall": recall_score(y_test, y_test_pred),
            "classification_report": classification_report(y_test, y_test_pred, output_dict=True),
            "confusion_matrix": confusion_matrix(y_test, y_test_pred),
            "prediction": y_test_pred,
        }
        
        # Append results
        results.append(metrics)
        
        # Display results in Streamlit
        st.write(f"### Results for {model_name}:")
        st.write(f"**Train Accuracy:** {metrics['train_accuracy']:.2f}")
        st.write(f"**Test Accuracy:** {metrics['test_accuracy']:.2f}")
        st.write(f"**F1 Score (Train):** {metrics['f1_score_train']:.2f}")
        st.write(f"**F1 Score (Test):** {metrics['f1_score_test']:.2f}")
        st.write(f"**Precision:** {metrics['precision']:.2f}")
        st.write(f"**Recall:** {metrics['recall']:.2f}")
        st.write("**Confusion Matrix:**")
        st.write(metrics["confusion_matrix"])
    
    return results
