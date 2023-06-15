import streamlit as st
import pandas as pd
import os
import ydata_profiling
from streamlit_pandas_profiling import st_profile_report




#Sidebar
with st.sidebar:
    st.image("Bild1.png")
    st.title("Auto Steam ML")
    Choice = st.radio("Choose one of the following options:", ["Upload your Data", "Data Report", "ML Model Analysis"])

#Check if some Data is already Existing
if os.path.exists("saved_data_file.csv"):
    data_df = pd.read_csv("saved_data_file.csv", index_col=None)

if Choice == "Upload your Data":
    st.title("Choose your Data")
    data_file = st.file_uploader("Uploda Data here!")
    if data_file:
        data_df = pd.read_csv(data_file, index_col=None)
        data_df.to_csv("saved_data_file.csv", index=None)
        st.dataframe(data_df)

if Choice == "Data Report":
    st.title("Get an automated ydata_profiling_report from your uploaded Data!")
    profile_report = data_df.profile_report()
    st_profile_report(profile_report)