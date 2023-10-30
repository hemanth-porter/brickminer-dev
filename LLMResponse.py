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
from config import model_path
from secret_keys import API_KEY
from streamlit_download import download_button
import pdb
import streamlit as st


from constants import complaints_modelling_prompt,positive_points_modelling_prompt,change_format_prompt,theme_finder_prompt,summarize_prompt


openai.api_key = API_KEY
from helper_utils import timer, retry


class LLMResponse():
        @retry
        def ask_gpt(self,system_instruction, list_of_reviews,return_type = None, is_eval = False, gpt4 = True ):
            if gpt4 == False:
                response = openai.ChatCompletion.create(
                    model='gpt-3.5-turbo',
                    messages=[
                        {"role": "system", "content": f"{system_instruction}"},
                        {"role": "user", "content": f"{list_of_reviews}"},
                    ])


                message = response.choices[0]['message']['content']

            elif gpt4 == True:
                response = openai.ChatCompletion.create(
                    model='gpt-4',
                    messages=[
                        {"role": "system", "content": f"{system_instruction}"},
                        {"role": "user", "content": f"{list_of_reviews}"},
                    ])

                message = response.choices[0]['message']['content']

            #This is only true for theme_finder
            #This is written because, sometimes we get apostrpohe because of which eval failes, 
            #Then this code reruns and gpt shall give one without apostraphe
            if is_eval :
                try:
                    if return_type == None:
                        return eval(message)
                    else:
                        if type(eval(message)) == return_type:
                            return eval(message)
                        else:
                            logging.ERROR(f"ask_gpt returned type is incorrect. Requested {return_type}, returned {type(eval(message))}")
                            raise TypeError(f"ask_gpt returned type is incorrect. Requested {return_type}, returned {type(eval(message))}")


                except SyntaxError as e:
                    prompt = system_instruction + list_of_reviews + message
                    continue_message = self.continue_gpt(prompt)

                return eval(message)               
                    
            return message        


        def continue_gpt(self,prompt, gpt4 = False ):
            if gpt4 == False:
                response = openai.ChatCompletion.create(
                    model='gpt-3.5-turbo',
                    messages=[
                        {"role": "system", "content": f"{prompt}"},
                        {"role": "user", "content": f"Continue"},
                    ])

                message = response.choices[0]['message']['content']
            elif gpt4 == True:
                response = openai.ChatCompletion.create(
                    model='gpt-4',
                    messages=[
                        {"role": "system", "content": f"{prompt}"},
                        {"role": "user", "content": f"Continue"},
                    ])

                message = response.choices[0]['message']['content']

            return message
        
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
                self.summary_progress_bar.progress(summary_percnt_done)
                self.summary_status_text.text(f'{summary_percnt_done}%') 

                self.summary_count = self.summary_count + 1

            except:
                self.summary_count = self.summary_count + 1
                return None
            return message
        
 
