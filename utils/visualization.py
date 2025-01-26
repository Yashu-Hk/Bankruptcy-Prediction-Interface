import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

def create_dashboard_visualizations(data):
    fig, axes = plt.subplots(3, 2, figsize=(12, 12))

    # Histogram of ROA Before Interest
    sns.histplot(data[" ROA(C) before interest and depreciation before interest"], ax=axes[0, 0])
    axes[0, 0].set_title("ROA Before Interest")
    
    # Boxplot of Debt Ratio
    sns.boxplot(x=data[" Debt ratio %"], ax=axes[0, 1])
    axes[0, 1].set_title("Debt Ratio")

    # Line plot for Net Value Per Share
    sns.lineplot(x=data.index, y=data[" Net Value Per Share (B)"], ax=axes[1, 0])
    axes[1, 0].set_title("Net Value Per Share")

    # Scatter plot for Growth Rates
    sns.scatterplot(x=data[" Total Asset Growth Rate"], y=data[" Operating Profit Growth Rate"], ax=axes[1, 1])
    axes[1, 1].set_title("Growth Rates")

    # Histogram for Net Profit Growth
    sns.histplot(data[" Continuous Net Profit Growth Rate"], ax=axes[2, 0])
    axes[2, 0].set_title("Net Profit Growth")

    st.pyplot(fig)
