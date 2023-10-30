import pandas as pd 
from tqdm import tqdm
from scipy.special import softmax
from config import model_path
from streamlit_download import download_button
import streamlit as st

import logging

class DataLoader():
    def __init__(self):
        
        self.data = None
        self.continue_pressed = False


    def load_data_via_upload(self):
        """
        Loads data from streamlit UI. Takes CSV Input 
        """
        uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

        if uploaded_file is not None:            
            df = pd.read_csv(uploaded_file)

            options = list(df.columns)
            col_name = st.selectbox('Pick a Column to Analyze', options)

            suggestions = df[col_name].dropna()
            self.data = pd.DataFrame(suggestions)
            self.data.columns = ['review']
            if st.button("Continue"):
                self.continue_pressed = True
                logging.info("File uploaded done")
                logging.info(f"Uploaded file name - {uploaded_file.name}")


                return self.data, self.continue_pressed  
            else:
                return self.data, self.continue_pressed  #Returns None
            
        else:
            st.write("No CSV file uploaded yet.")
            return self.data, self.continue_pressed #Returns None



