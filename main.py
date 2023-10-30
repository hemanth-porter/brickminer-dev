import logging
from secret_keys import aws_access_key_id,aws_secret_access_key

my_logger = logging.getLogger()

import helper_utils
import boto3

import time
import numpy as np
import pandas as pd  
import streamlit as st

from config import model_path
from streamlit_download import download_button

from DataLoader import DataLoader
from SentimentAnalyser import SentimentAnalyser
from TopicModelling import TopicModelling 
from TopicProcessing import TopicProcessing
from ThemeModelling import ThemeModelling


st.set_page_config(
    page_title="Brick Miner",
    page_icon="🔍"
)


if not my_logger.handlers:
    my_logger.setLevel(logging.INFO)
    log_file_time = time.time()

    #Logs Filename, Time, LoggingLevel, Message, ModuleName, LineNo
    formatter = logging.Formatter("%(name)s : %(asctime)s : %(levelname)s : %(message)s : %(module)s : %(lineno)d")
    file_handler = logging.FileHandler(f'BrickMiner_{log_file_time}.log')
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

        self.positive, self.negative, self.neutral = sentiment_analyser_object.get_sentiment()

        sentiment_analyser_object.show_sentiment_metrics() #Shows sentiment splitup
        my_logger.info('Sentiment Metrics Displayed')
        print("Sentiment Metrics Displayed")

    def run_topic_modelling(self):

        topic_modelling_object = TopicModelling()
        
        #Negative
        try:
            if len(self.negative)>0:                
                self.topic_data_neg = topic_modelling_object.get_analysis(df = self.negative,what = 'complaints')
            
        except:
            my_logger.error("Failed TopicModelling for Negative")

        #Positive
        try:
            if len(self.positive)>0:                
                self.topic_data_pos = topic_modelling_object.get_analysis(df = self.positive,what = 'positives')
            
        except:
            my_logger.error("Failed TopicModelling for Positive")

    def run_topic_processing(self):        

        def run_topic_processing_helper(df,what):
            topic_processor_object = TopicProcessing()

            reviews_summary = topic_processor_object.get_topic_priority_and_percentage(df)

            if what == 'complaints':
                self.complaints_topics = reviews_summary                
            elif what == 'positives':
                self.positives_topics = reviews_summary                


        #Negative
        if len(self.negative) >0:    
            
            run_topic_processing_helper(df = self.negative,what = 'complaints')
            logging.info("Got Summary and Priority Percentage for Negative")


        #Positive
        if len(self.positive) >0:    
            
            run_topic_processing_helper(df = self.positive,what = 'positives')
            logging.info("Got Summary and Priority Percentage for Positive")

    def run_theme_modelling(self,display = False):
        """
        Runs theme finder on Topics to find the similar topics and cluster into themes.
        Also, Gives a download button to download theme-wise results
        """
        theme_modelling_object = ThemeModelling()

        if len(self.topic_data_neg) >0 :
            # Negative
            
            self.theme_data_neg = theme_modelling_object.themify(self.topic_data_neg)

            if display:
                theme_modelling_object.display_theme_data(self.theme_data_neg)
            logging.info("Theme Modelling done for Negative")

        if len(self.topic_data_pos) >0 :
            # Positive
            
            self.theme_data_pos = theme_modelling_object.themify(self.topic_data_pos)

            if display:
                theme_modelling_object.display_theme_data(self.theme_data_pos)
            logging.info("Theme Modelling done for Positive")

 

    def structure_output(self):

        def structure_output_helper(theme_data,topic_data):
            theme_data['Topics_list']= theme_data['TopicsDataframe'].apply(lambda x: list(x['topic']))
            theme_data_exploded = theme_data.explode('Topics_list').copy()
            topic_subtopic = topic_data.merge(theme_data_exploded, left_on = 'topics',right_on = 'Topics_list', how = "outer")
            topic_subtopic_final = topic_subtopic[['Theme','n_Reviews','topics','n_reviews_for_this_topic','raw_reviews','summaried_reviews']].copy()
            topic_subtopic_final.sort_values(by = ['n_Reviews','Theme'],ascending = False, inplace = True)

            topic_subtopic_final.columns = ['Topic name','Number of reviews for theme','Topic name','Number of reviews for topic','Original reviews','Summary']

            return topic_subtopic_final        
        
        if len(self.topic_data_neg) >0 :
            negative_output = structure_output_helper(self.theme_data_neg,self.complaints_topics)

        if len(self.topic_data_pos) >0 : 
            positive_output = structure_output_helper(self.theme_data_pos,self.positives_topics)

        negative_output_str = download_button(negative_output,'negative_output.csv','Download results for complaints')
        st.markdown(negative_output_str, unsafe_allow_html=True) 

        positive_output_str = download_button(positive_output,'positive_output.csv','Download results for positives')
        st.markdown(positive_output_str, unsafe_allow_html=True)         
    
    def push_to_s3(self):
        session = boto3.Session(
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        )
        s3 = session.resource('s3')
        s3.meta.client.upload_file(Filename=f'BrickMiner_{log_file_time}.log',
                                    Bucket='brickminer-models',
                                    Key=f'application-logs/BrickMiner_{log_file_time}.log')



    def run(self):
        analysis_completed = st.session_state.get("analysis_completed", False)
        self.load_data()
        if self.data is not None :
            my_logger.info(f"Uploaded file rows - {len(self.data)}")

            memory_usage_bytes = self.data.memory_usage().sum()      
            memory_usage_mb = memory_usage_bytes / (1024 * 1024) # Convert from bytes to megabytes

            my_logger.info(f"Uploaded file size - {memory_usage_mb} MB")                            

        if ( self.data is not None ) & ( self.continue_pressed ):
            warning_container = st.empty()
            warning_container.warning('Analysing data.. This might take a while!', icon="⚠️")

            with st.spinner('Step 1/4 in progress'):
                self.get_sentiment_data_and_metrics()

            with st.spinner('Step 2/4 in progress'):
                self.run_topic_modelling()
            
            with st.spinner('Step 3/4 in progress'):
                self.run_topic_processing()
            
            with st.spinner('Step 4/4 in progress'):
                self.run_theme_modelling(display=False)
                self.structure_output()
                self.push_to_s3()
            
            warning_container.success("✅ Analysis completed successfully!")

if __name__ == '__main__':
    try:
        BMObject = BrickMiner()
        BMObject.run()
    except Exception as e:
        my_logger.exception("BrickMiner failed with error {e}")




   




    









        





