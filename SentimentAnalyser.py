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

import string 
from tqdm import tqdm
import re 
import gensim 
from gensim import corpora, models   


from scipy.special import softmax
import pdb
import streamlit as st

import nltk 
from nltk.stem import WordNetLemmatizer  
from nltk.corpus import stopwords  
from nltk.tokenize import word_tokenize 
from nltk.stem import PorterStemmer 
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet') 

import logging


from config import model_path
from config import API_KEY
from streamlit_download import download_button
from helper_utils import timer


class SentimentAnalyser():
    def __init__(self,data,n_sentiment_labels):
        
        self.model_path = model_path
        self.positive = None
        self.negative = None 
        self.neutral = None

        self.tokenizer = None

        #Below Shall be passed during initialising the class
        self.data = data
        self.n_sentiment_labels = n_sentiment_labels
        self.tokenizer, self.model, self.autoconfig = self.load_sentiment_model()

    @st.cache_resource
    def load_sentiment_model(_self):
        tokenizer = AutoTokenizer.from_pretrained(_self.model_path)
        model = AutoModelForSequenceClassification.from_pretrained(_self.model_path)
        autoconfig =  AutoConfig.from_pretrained(_self.model_path)    
        
        return tokenizer, model, autoconfig
        

    @staticmethod
    def preprocess(text):
        """
        
        """
        text = text.translate(str.maketrans("", "", string.punctuation))
        text = text.lower()
        words = word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        words = [word for word in words if word not in stop_words]
        lemmatizer = WordNetLemmatizer()
        words = [lemmatizer.lemmatize(word) for word in words]
        text = " ".join(words)
        return text


    def get_sentiment_helper(self,text):

        labels = self.n_sentiment_labels

        text = self.preprocess(text)
        encoded_input = self.tokenizer(text, return_tensors='pt')
        output = self.model(**encoded_input)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)

        ranking = np.argsort(scores)
        ranking = ranking[::-1]
        details_text = ''
        
        for i in range(scores.shape[0]):
            l = self.autoconfig.id2label[ranking[i]]
            s = scores[ranking[i]]
            details_text = details_text + f"{l}. {np.round(float(s), 4)} "

        if labels == 3:
            if scores.argmax() == 0:
                return "Negative"            
            if scores.argmax() == 1:
                return "Neutral"
            elif scores.argmax() == 2:
                return "Positive" 
            
        elif labels == 2:
            if scores.argmax() == 1:
                return "Postive"
            elif scores.argmax() == 0:
                return "Negative"
                

    def sentiment_splitter(self):

        labels = self.n_sentiment_labels
        data = self.data

        if labels == 2 :
            positive = data[data['sentiment'] == 'Postive'].copy()
            negative = data[data['sentiment'] == 'Negative'].copy()
            return positive, negative, None
        
        elif labels == 3:
            positive = data[data['sentiment'] == 'Postive'].copy()
            negative = data[data['sentiment'] == 'Negative'].copy()
            neutral = data[data['sentiment'] == 'Neutral'].copy()
            return positive, negative, neutral
        
    @timer
    def get_sentiment(self):        
        self.load_sentiment_model()
        logging.info('Sentiment Model Loaded')

        self.data['sentiment'] = self.data.review.apply(lambda x: self.get_sentiment_helper(x) )
        self.positive, self.negative, self.neutral = self.sentiment_splitter()

        logging.info('Sentiment Split Done')
        return  self.positive, self.negative, self.neutral
        

    def show_sentiment_metrics_helper(self,total,pos_len,neg_len,neutral_len):

        pos_percent = round(pos_len*100/total,2)
        neg_percent = round(neg_len*100/total,2)
        neutral_percent = round(neutral_len*100/total,2)
        cols = st.columns(self.n_sentiment_labels + 1)
        cols[0].metric("Total reviews (After removing blanks)", str(total))
        cols[1].metric("Positive reviews", str(pos_len)+f" ( {pos_percent}% )")
        cols[2].metric("Negative reviews", str(neg_len)+f" ( {neg_percent}% )")
        
        if self.n_sentiment_labels == 3:
            cols[3].metric("Neutral reviews", str(neutral_len)+f" ( {neutral_percent}% )")

    def show_sentiment_metrics(self):

        if self.neutral == None:
            neutral_len = 0
        else:
            neutral_len = len(self.neutral)

        logging.info(f"Length of non-null data - {len(self.data)}")
        logging.info(f"Length of positive data - {len(self.positive)}")
        logging.info(f"Length of negative data - {len(self.negative)}")
        
        self.show_sentiment_metrics_helper(len(self.data),len(self.positive),len(self.negative),neutral_len)
    
