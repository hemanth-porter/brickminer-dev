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
from constants import theme_finder_prompt
from LLMResponse import LLMResponse
import logging

class ThemeModelling():

    def validate_theme_topics(self,my_topics_list,theme_repsose_dict):

        #Fetch the topics from the reponse
        themes_names = list(theme_repsose_dict.keys())

        topics_list_from_themes = []
        for i in range(len(themes_names)):
            topics_list_from_themes.extend(theme_repsose_dict[themes_names[i]])

        #Validate with the topics that we have sent
        missed_topics = [x for x in my_topics_list if x not in topics_list_from_themes]

        return missed_topics

    def theme_finder(self, my_topics_list):   

        llm_response_object = LLMResponse()     
        
        theme_repsose_dict = llm_response_object.ask_gpt(theme_finder_prompt, my_topics_list, is_eval = True, gpt4= True)
        
        #Checks if there are any topics that are missed from what we have given in the response_dict
        missed_topics = self.validate_theme_topics(my_topics_list, theme_repsose_dict)

        print("Missed topics: ", missed_topics)

        if len(missed_topics) > 3:
            missed_topics_dict = self.theme_finder(missed_topics)
            new_topics_dict = {}

            for key, value in missed_topics_dict.items():
                if key in theme_repsose_dict:
                    new_topics_dict[key + "_new"] = value
                else:
                    theme_repsose_dict[key] = value

            theme_repsose_dict.update(new_topics_dict)

        return theme_repsose_dict
    
    def get_theme_contri_metrics(self,df,theme_name,theme_topics_list):


        reviews_for_theme = df[df['exploded_topic'].isin(theme_topics_list)].copy()

        #no of unique reviews for the current theme
        unique_reviews_under_theme = reviews_for_theme['review'].nunique()

        # No of times Topic under this theme are talked ( Non - unique review)
        total_reviews_under_theme = len(reviews_for_theme['review'])

        #Percentage contri of each topic under a theme
        topic_contri_df = pd.DataFrame(reviews_for_theme['exploded_topic'].value_counts()*100/len(reviews_for_theme))
        topic_contri_df = topic_contri_df.reset_index()
        topic_contri_df.columns = ['topic','topic_contri_for_theme']

        return theme_name, unique_reviews_under_theme,topic_contri_df
    
    def get_topic_explode(self, topic_list):
        try:
            return eval(topic_list)
        except:
            return []

    def themify(self,df):
        """
        df = analsyed_data_neg / analsyed_data_pos
        """

        df['exploded_topic'] = df['topic'].apply(lambda x: self.get_topic_explode(x))

        df_exploded = df.explode("exploded_topic")
        list_of_unique_topics = list(df_exploded['exploded_topic'].unique())

        theme_repsose_dict = self.theme_finder(list_of_unique_topics)

        themes_list = list(theme_repsose_dict.keys())

        theme_results = []
        for i in range(len(themes_list)):
            current_theme_results = self.get_theme_contri_metrics(df_exploded, themes_list[i], theme_repsose_dict[themes_list[i]])

            theme_results.append(current_theme_results)

        theme_results_df = pd.DataFrame(theme_results)
        theme_results_df.columns =  ['Theme','n_Reviews','TopicsDataframe']

        theme_results_df.sort_values(by='n_Reviews', ascending=False, inplace = True)

        return theme_results_df
    
    def show_theme_results(self,individual_theme_results_list):
        theme_name, unique_reviews_under_theme,total_reviews_under_theme,topic_contri_df = individual_theme_results_list
        st.write("Theme Name: ",theme_name)
        st.write("Unique Reviews: ",unique_reviews_under_theme)
        st.write("Total Reviews: ",total_reviews_under_theme)
        st.write(topic_contri_df)


    def display_theme_data(self, df):        

        df.sort_values(by = 'n_Reviews' )

        # Display each theme
        for index, row in df.iterrows():
            theme = row['Theme']
            reviews = row['n_Reviews']
            dataframe = row['TopicsDataframe']

            # Display theme name and number of reviews
            button_label = f"{theme} (Reviews: {reviews})"
            with st.expander(button_label, expanded=False):
                # Display dataframe with topics and contributions
                st.table(dataframe)

            # Add some spacing
            st.write('\n')
