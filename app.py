from operator import index
import streamlit as st
import plotly.express as px
from pycaret.regression import setup, compare_models, pull, save_model, load_model
import pandas_profiling
import pandas as pd
from streamlit_pandas_profiling import st_profile_report
import os 


from PIL import Image

if os.path.exists('./dataset.csv'): 
    df = pd.read_csv('dataset.csv', index_col=None)
    



original_title = '<p style="font-family:cursive; color:lightskyblue; font-size: 60px;"><strong>Welcome to Auto ML</strong></p>'
st.markdown(original_title, unsafe_allow_html=True)

with st.sidebar: 
    st.image("https://www.trackmatrix.com/wp-content/uploads/2022/09/TM_homepage_top-banner3_H500px.png.webp")
    st.title("Auto ML")
    choice = st.radio("Navigation", ["Upload Data","Analysis","ML Modelling", "Download Model"])
    st.info("This application helps you Upload, build, explore your data and build Machine Learning Models.")

if choice == "Upload Data":
    st.title("Upload Your Dataset")
    file = st.file_uploader("Upload Your Dataset")
    if file: 
        df = pd.read_csv(file, index_col=None)
        df.to_csv('dataset.csv', index=None)
        st.dataframe(df)

if choice == "Analysis": 
    st.title("Exploratory Data Analysis")
    profile_df = df.profile_report()
    st_profile_report(profile_df)

if choice == "ML Modelling": 
    chosen_target = st.selectbox('Choose the Target Column', df.columns)
    if st.button('Run Modelling'): 
        setup(df, target=chosen_target)
        setup_df = pull()
        st.dataframe(setup_df)
        best_model = compare_models()
        compare_df = pull()
        st.dataframe(compare_df)
        save_model(best_model, 'best_model')

if choice == "Download Model": 
    with open('best_model.pkl', 'rb') as f: 
        st.download_button('Download Model', f, file_name="best_model.pkl")
        

