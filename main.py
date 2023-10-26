import logging
import functools
import time
import numpy as np
from transformers import pipeline
from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
import openai
import string
import pandas as pd 
import os  
from tqdm import tqdm
import nltk 
from nltk.stem import WordNetLemmatizer  
from nltk.corpus import stopwords  
import string 
from tqdm import tqdm
import re 
import gensim 
from gensim import corpora, models   
from nltk.tokenize import word_tokenize 
from nltk.stem import PorterStemmer 
from scipy.special import softmax
import pdb
import streamlit as st

from config import model_path
from config import API_KEY
from streamlit_download import download_button

from DataLoader import DataLoader
from SentimentAnalyser import SentimentAnalyser
from TopicModelling import TopicModelling 
from TopicProcessing import TopicProcessing
from ThemeModelling import ThemeModelling


my_logger = logging.getLogger()

if not my_logger.handlers:
    my_logger.setLevel(logging.INFO)

    #Logs Filename, Time, LoggingLevel, Message, ModuleName, LineNo
    formatter = logging.Formatter("%(name)s : %(asctime)s : %(levelname)s : %(message)s : %(module)s : %(lineno)d")
    file_handler = logging.FileHandler(f'BrickMiner_{time.time()}.log')
    file_handler.setFormatter(formatter)
    my_logger.addHandler(file_handler)



class BrickMiner():
    def __init__(self):
        
        self.data = None
        self.continue_pressed = False
        self.load_type = 'CSV Upload'

        self.positive = None
        self.negative = None
        self.neutral = None

        self.n_sentiment_labels = 2

        self.topic_data_pos = pd.DataFrame()
        self.topic_data_neg = pd.DataFrame()

    def load_data(self):
        data_loader_object = DataLoader()
        if self.load_type == 'CSV Upload':
            self.data, self.continue_pressed =  data_loader_object.load_data_via_upload()

    def get_sentiment_data_and_metrics(self):
        sentiment_analyser_object =  SentimentAnalyser(self.data, self.n_sentiment_labels)

        st.info("Fetching Sentiment for all feedback.. Please wait..")
        self.positive, self.negative, self.neutral = sentiment_analyser_object.get_sentiment()

        sentiment_analyser_object.show_sentiment_metrics() #Shows sentiment splitup
        my_logger.info('Sentiment Metrics Displayed')
        print("Sentiment Metrics Displayed")

    def run_topic_modelling(self):

        topic_modelling_object = TopicModelling()
        
        #Negative
        try:
            if len(self.negative)>0:
                st.info("Fetching Topics for negative feedback.. Please wait..")
                self.topic_data_neg = topic_modelling_object.get_analysis(df = self.negative,what = 'complaints')
            # st.write(self.topic_data_neg)
        except:
            my_logger.error("Failed TopicModelling for Negative")

        #Positive
        try:
            if len(self.positive)>0:
                st.info("Fetching Topics for positive feedback.. Please wait..")
                self.topic_data_pos = topic_modelling_object.get_analysis(df = self.positive,what = 'positives')
            # st.write(self.topic_data_pos)
        except:
            my_logger.error("Failed TopicModelling for Positive")

    def run_topic_processing(self):        

        def run_topic_processing_helper(df,what):
            topic_processor_object = TopicProcessing()

            reviews_summary = topic_processor_object.get_topic_priority_and_percentage(df)

            review_summaried_button = download_button(reviews_summary,f'topicwise_data_{what}.csv',f'Download topic wise results for {what}')
            st.markdown(review_summaried_button, unsafe_allow_html=True)     

        #Negative
        if len(self.negative) >0:    
            st.info("Fetching summary for negative feedback topics.. Please wait..")
            run_topic_processing_helper(df = self.negative,what = 'complaints')
            logging.info("Got Summary and Priority Percentage for Negative")

        #Positive
        if len(self.positive) >0:    
            st.info("Fetching summary for positive feedback topics.. Please wait..")
            run_topic_processing_helper(df = self.positive,what = 'positives')
            logging.info("Got Summary and Priority Percentage for Positive")

    def run_theme_modelling(self):
        """
        Runs theme finder on Topics to find the similar topics and cluster into themes.
        Also, Gives a download button to download theme-wise results
        """
        theme_modelling_object = ThemeModelling()

        if len(self.topic_data_neg) >0 :
            # Negative
            st.info("Fetching Themes for negative feedback.. Please wait..")
            self.theme_data_neg = theme_modelling_object.themify(self.topic_data_neg)
            neg_theme_results_df_button_str = download_button(self.theme_data_neg,'themewise_complaints.csv','Download theme wise results for complaints')
            st.markdown(neg_theme_results_df_button_str, unsafe_allow_html=True) 
            theme_modelling_object.display_theme_data(self.theme_data_neg)
            logging.info("Theme Modelling done for Negative")

        if len(self.topic_data_pos) >0 :
            # Positive
            st.info("Fetching Themes for positive feedback.. Please wait..")
            self.theme_data_pos = theme_modelling_object.themify(self.topic_data_pos)
            pos_theme_results_df_button_str = download_button(self.theme_data_pos,'themewise_positive.csv','Download theme wise results for positives')
            st.markdown(pos_theme_results_df_button_str, unsafe_allow_html=True)
            theme_modelling_object.display_theme_data(self.theme_data_pos)
            logging.info("Theme Modelling done for Positive")


    def run(self):

        self.load_data()
        if self.data is not None :
            my_logger.info(f"Uploaded file rows - {len(self.data)}")

            memory_usage_bytes = self.data.memory_usage().sum()      
            memory_usage_mb = memory_usage_bytes / (1024 * 1024) # Convert from bytes to megabytes

            my_logger.info(f"Uploaded file size - {memory_usage_mb} MB")                

        if ( self.data is not None ) & ( self.continue_pressed ):
            self.get_sentiment_data_and_metrics()
            self.run_topic_modelling()
            self.run_topic_processing()
            self.run_theme_modelling()

            

if __name__ == '__main__':
    try:
        BMObject = BrickMiner()
        BMObject.run()
    except Exception as e:
        my_logger.exception("BrickMiner failed with error {e}")




   




    









        





