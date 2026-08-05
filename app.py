import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="HR Employee Attrition Dashboard",
    page_icon="👨‍💼",
    layout="wide"
)

st.title("👨‍💼 HR Employee Attrition Prediction System")

st.markdown("---")

st.write("""
### Project Overview

This dashboard predicts whether an employee is likely to leave the company.

### Features

- 📊 Exploratory Data Analysis
- 🤖 Employee Attrition Prediction
- 📈 Model Performance
- 📉 Feature Importance
- 📋 HR Business Insights

Use the sidebar to navigate between pages.
""")

df = pd.read_csv("clean_hr_data.csv")

st.markdown("---")

col1,col2,col3,col4 = st.columns(4)

col1.metric("Employees",len(df))
col2.metric("Departments",df["Department"].nunique())
col3.metric("Job Roles",df["JobRole"].nunique())
col4.metric("Attrition Rate",
            f"{round(df['Attrition'].mean()*100,2)}%")

st.markdown("---")

st.subheader("Dataset Preview")

st.dataframe(df.head())