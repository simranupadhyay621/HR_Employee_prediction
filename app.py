import streamlit as st
import pandas as pd
import pickle

model = pickle.load(open("model.pkl","rb"))
scaler = pickle.load(open("scaler.pkl","rb"))
encoders = pickle.load(open("encoders.pkl","rb"))

df = pd.read_csv("Human_Resources.csv")

st.set_page_config(
    page_title="HR Attrition Prediction",
    layout="wide"
)

st.title("HR Employee Attrition Prediction")

st.write("Enter employee information")

feature_df = df.drop(
    ["Attrition",
     "EmployeeCount",
     "EmployeeNumber",
     "Over18",
     "StandardHours"],
     axis=1
)

user_data = {}

for col in feature_df.columns:

    if feature_df[col].dtype=="object":
        user_data[col]=st.selectbox(
            col,
            feature_df[col].unique()
        )

    else:
        user_data[col]=st.number_input(
            col,
            value=float(feature_df[col].median())
        )

input_df = pd.DataFrame([user_data])

for col in input_df.select_dtypes(include="object").columns:
    input_df[col]=encoders[col].transform(input_df[col])

scaled = scaler.transform(input_df)

prediction = model.predict(scaled)

if st.button("Predict"):

    if prediction[0]==1:
        st.error("Employee is likely to leave.")
    else:
        st.success("Employee is likely to stay.")
