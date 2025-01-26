import streamlit as st
import pandas as pd
import joblib
from utils.visualization import create_dashboard_visualizations
from utils.pdf_generator import generate_pdf_report
from utils.preprocessing import preprocess_data, run_ml_models

# Define the models directory (relative path)
models_dir = "models"

# Load trained ML models
models = {
    "Logistic Regression": joblib.load(f"{models_dir}/logistic_regression.pkl"),
    "Decision Tree": joblib.load(f"{models_dir}/decision_tree.pkl"),
    "Naive Bayes": joblib.load(f"{models_dir}/naive_bayes.pkl"),
    "Random Forest": joblib.load(f"{models_dir}/random_forest.pkl"),
    "SVM": joblib.load(f"{models_dir}/svm.pkl"),
}

# Streamlit App
st.set_page_config(page_title="Bankruptcy Prediction App", layout="wide")

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Dashboard"])

# Home Page
if page == "Home":
    st.title("Bankruptcy Prediction App")
    st.header("Upload Your Dataset")
    
    uploaded_file = st.file_uploader("Upload a CSV or TXT file with 95 columns", type=["csv", "txt"])
    
    if uploaded_file:
        st.success("File uploaded successfully!")
        data = pd.read_csv(uploaded_file)
        st.dataframe(data.head())
        
        if st.button("Predict"):
            with st.spinner("Processing your data..."):
                # Preprocess the dataset
                X, y = preprocess_data(data)

                if X.empty or y.empty:
                    st.error("The dataset is empty. Please upload a valid file with data.")
                    st.stop() 

                # Run the ML models
                results = run_ml_models(models, X, y)

                # Identify the best-performing model
                best_model = max(results, key=lambda x: x["test_accuracy"])
                best_model_name = best_model["name"]

                st.success(f"Prediction completed! Best model: {best_model_name} "
                           f"(Test Accuracy: {best_model['test_accuracy']:.2f})")

                # Store results in session state
                st.session_state["results"] = results
                st.session_state["data"] = data
                st.session_state["prediction"] = best_model["prediction"]

# Dashboard Page
elif page == "Dashboard":
    st.title("Company Financial Dashboard")

    if "results" in st.session_state:
        data = st.session_state["data"]
        predictions = st.session_state["prediction"]

        # Display financial visualizations
        create_dashboard_visualizations(data)

        # Display bankruptcy prediction status
        bankruptcy_status = "Bankrupt" if predictions[0] == 1 else "Not Bankrupt"
        st.markdown(f"### Bankruptcy Prediction: **{bankruptcy_status}**")

        # Generate PDF report
        if st.button("Generate PDF Report"):
            generate_pdf_report(data, st.session_state["results"], bankruptcy_status)
            st.success("PDF report generated! Check your downloads folder.")

    else:
        st.warning("Please upload and predict data on the Home page first.")
