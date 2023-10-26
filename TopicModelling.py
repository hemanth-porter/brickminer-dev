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

import logging



from config import model_path
from config import API_KEY
from streamlit_download import download_button
from helper_utils import retry, timer
from LLMResponse import LLMResponse
from constants import complaints_modelling_prompt, positive_points_modelling_prompt, change_format_prompt, summarize_prompt


class TopicModelling():
    def __init__(self):
        self.complaints_list = []
        self.positives_list = []

        self.complaints_failed_indx = []
        self.positives_failed_indx = []
        self.llm_response_object = LLMResponse()

        self.summary_count = 0
                
    # @retry
    def topic_modeling_gpt(self, each_review,what):
        

        if what == 'complaints':
            try:
                results = self.llm_response_object.ask_gpt(complaints_modelling_prompt.format(self.complaints_list), each_review)            
                return results
            except:
                print(f"Faled for {each_review}")
                logging.error(f"Complaints Topic Modelling Failed for {each_review}")

        elif what == 'positives':
            return self.llm_response_object.ask_gpt(positive_points_modelling_prompt.format(self.positives_list), each_review)
    
    # @retry
    def handle_failed_gpt_modeling(self,failed_str,actual_review):
        """
        Return if the len of review < 25 ( Tunable )
        Else, Tries running formatter 
        """

        if len(actual_review) < 25:
            return actual_review
        
        else:
            return self.llm_response_object.ask_gpt(change_format_prompt,failed_str)
        

    def add_point_to_df(self,df,indx,return_string):
        df.iloc[indx,df.columns.get_loc('topic')] = return_string
        
    def add_new_points_to_list(self, return_string, what):    
        """
        return_string = 'returned complaints/positive points from gpt'
        what = 'complaints/positives'
        """
        return_list = eval(return_string)

        if what == 'complaints':
            for each_new_complaint in return_list:
                if each_new_complaint not in self.complaints_list:
                    self.complaints_list.append(each_new_complaint)

        elif what == 'positives':
            for each_positive_point in return_list:
                if each_positive_point not in self.positives_list:
                    self.positives_list.append(each_positive_point)


    def add_actual_review_to_df_and_list(self, df,indx):

        actual_review = f"['{df['review'].iloc[indx]}']"
        df.iloc[indx,df.columns.get_loc('topic')] = actual_review
        self.complaints_list.append(actual_review)    

    @timer
    def get_analysis(self, df, what  ):
        
        failed_indx = []

        status_text = st.text('0%')
        progress_bar = st.progress(0)

        df['error'] = np.nan
        df['second_error'] = np.nan

        df['topic'] = np.nan

        for indx in range(len(df['review'])):
            
            percnt_done = int(indx *100 /(len(df['review'])-1))
            progress_bar.progress(percnt_done)
            status_text.text(f'{percnt_done}%') 

            each_review = df['review'].iloc[indx]
            return_string = self.topic_modeling_gpt(each_review,what)
            add_points_to_list = False
            try:
                self.add_new_points_to_list(return_string,what)
                add_points_to_list = True
                self.add_point_to_df(df,indx,return_string)

            except Exception as e:

                #Add logic to handle if it is string error then make it a list input : 'no complaints' output : ['no complaints']

                logging.error(e)
                logging.info(f"Faied for {indx}: {return_string}")
                failed_indx.append(indx)
                df.iloc[indx,df.columns.get_loc('error')] = e
            
                corrected_str = self.handle_failed_gpt_modeling(return_string,each_review)
                try:
                    #checks if the corrected string can be added to list and df"
                    self.add_new_points_to_list(corrected_str,what)
                    self.add_point_to_df(df,indx,corrected_str)
                except Exception as e2:
                    logging.error(f"Failed for second time with {e2}")
                    logging.info(f"Failed for second time deets. Return String : {return_string}, Corrected String : {corrected_str}")
                    df.iloc[indx,df.columns.get_loc('second_error')] = e2
            
        if what == "complaints":
            self.complaints_failed_indx = failed_indx
            logging.info("Topic Modelling done for Negative")

        elif what == "positives":
            self.positives_failed_indx = failed_indx
            logging.info("Topic Modelling done for Positive")

        return df
    
    


    
        






