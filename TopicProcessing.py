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
from streamlit_download import download_button
from helper_utils import retry, timer
from LLMResponse import LLMResponse
from constants import complaints_modelling_prompt, positive_points_modelling_prompt, change_format_prompt, summarize_prompt


class TopicProcessing():
        def __init__(self) -> None:
               self.summary_count = 0


        def get_summary( self, df):

                try:

                    list_of_reviews = df['review_concat']
                    topic_name = df['exploded_topic']

                    response = openai.ChatCompletion.create(
                        model='gpt-3.5-turbo',
                        messages=[
                            {"role": "system", "content": summarize_prompt},
                            {"role": "user", "content": f"List of reviews : {list_of_reviews}, Topic: {topic_name}"},
                        ])

                    message = response.choices[0]['message']['content']

                    summary_percnt_done = int(self.summary_count*100/self.summary_end)
                    # self.summary_progress_bar.progress(summary_percnt_done)
                    # self.summary_status_text.text(f'{summary_percnt_done}%') 

                    self.summary_count = self.summary_count + 1

                except:
                    self.summary_count = self.summary_count + 1
                    return None
                return message
                
        def get_topic_priority_and_percentage(self,df):

                df['exploded_topic'] = df['topic'].apply(lambda x: self.get_topic_explode(x))
                df = df.explode("exploded_topic").copy()
                df_grouped = df.groupby(['exploded_topic'],as_index = False).agg({"sentiment":'count',
                                                                                "review":'unique'
                                                                                })

                df_grouped.sort_values(by = 'sentiment',ascending = False,inplace = True)                

                df_grouped['topic_percentage'] = df_grouped['sentiment'].apply(lambda x: x*100/df['review'].nunique())
                df_grouped['review_concat'] = df_grouped['review'].apply(lambda x: ",".join(x))

                # st.write("Please wait fetching summaries of each topic..")
                # self.summary_status_text = st.text('0%')
                # self.summary_progress_bar = st.progress(0)
                
                self.summary_end = len(df_grouped)-1
                df_grouped['summaried_reviews'] = df_grouped.apply(lambda x: self.get_summary(x), axis = 1)

                return_df = df_grouped[['exploded_topic','sentiment', 'topic_percentage','review','summaried_reviews']]

                return_df.columns = ['topics','n_reviews_for_this_topic','topic_percentage','raw_reviews','summaried_reviews']

                return return_df


        def get_topic_explode(self, topic_list):
            try:
                return eval(topic_list)
            except:
                return []    
            


