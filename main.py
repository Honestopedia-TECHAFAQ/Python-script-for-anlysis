import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tkinter import Tk, filedialog
import streamlit as st

def load_data(file_path):
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        st.error(f"File not found: {file_path}")
        st.stop()

def main():
    st.title("Interactive Data Analysis with Streamlit")

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if not uploaded_file:
        st.info("Please upload a CSV file.")
        st.stop()

    # Load the dataset
    data = load_data(uploaded_file)

    # Display the first few rows of the dataset
    st.subheader("Dataset Preview")
    st.write(data.head())

    # Basic statistics
    st.subheader("Basic Statistics")
    st.write(data.describe())

    # Check for missing values
    st.subheader("Missing Values")
    st.write(data.isnull().sum())

    # Visualize the distribution of numeric columns
    st.subheader("Numeric Columns Distribution")
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    for column in numeric_columns:
        st.write(f"### {column} Distribution")
        fig, ax = plt.subplots()
        sns.histplot(data[column].dropna(), kde=True, ax=ax)
        st.pyplot(fig)

    # Visualize count plots for categorical columns
    st.subheader("Categorical Columns Count")
    categorical_columns = data.select_dtypes(include=['object']).columns
    for column in categorical_columns:
        st.write(f"### {column} Count")
        fig, ax = plt.subplots()
        sns.countplot(x=column, data=data, ax=ax)
        st.pyplot(fig)

    # Correlation matrix
    st.subheader("Correlation Matrix")
    fig, ax = plt.subplots()
    correlation_matrix = data.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

if __name__ == "__main__":
    main()
