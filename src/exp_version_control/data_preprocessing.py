import numpy as np
import pandas as pd
import re
import os
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# fetch the data from data/raw
train_data = pd.read_csv('data/raw/train.csv')
test_data = pd.read_csv('data/raw/test.csv')


# transform the data
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download("punkt")
nltk.download('punkt_tab')

def lemmatization(text):
    lemmatizer = WordNetLemmatizer()
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(tokens)

def remove_stop_words(text):
    stop_words = set(stopwords.words('english'))
    tokens = nltk.word_tokenize(text)
    tokens = [token for token in tokens if token.lower() not in stop_words]
    return ' '.join(tokens)


def remove_numbers(text):
    return re.sub(r'\d+', '', text)


def lowercase_text(text):
    ## Convert text to lowercase
    return text.lower()


def remove_punctuation(text):
    ## Remove punctuation
    return text.translate(str.maketrans('', '', string.punctuation))


def remove_urls(text):
    ## Remove URLs
    return re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

def remove_small_sentences(df, min_length=3):
    ## Remove sentences shorter than min_length
    df['content'] = df['content'].apply(lambda x: ''.join(x.split()) if len(x.split()) >= min_length else '')
    
    return df[df['content'].str.strip().astype(bool)]  # Remove empty rows

def normalize_text(df):
    ## Normalize contents of the DataFrame
    df['content'] = df['content'].apply(lowercase_text)
    df['content'] = df['content'].apply(remove_numbers)
    df['content'] = df['content'].apply(remove_punctuation)
    df['content'] = df['content'].apply(remove_urls)
    df['content'] = df['content'].apply(remove_stop_words)
    df['content'] = df['content'].apply(lemmatization)
    return df

def normalize_sentence(sentence):
    ## Normalize a single sentence
    sentence = lowercase_text(sentence)
    sentence = remove_numbers(sentence)
    sentence = remove_punctuation(sentence)
    sentence = remove_urls(sentence)
    sentence = remove_stop_words(sentence)
    sentence = lemmatization(sentence)
    return sentence

train_processed_data = normalize_text(train_data)
test_processed_data = normalize_text(test_data)


# store the data inside data/processed
data_path = os.path.join("data", "processed")
os.makedirs(data_path)

train_processed_data.to_csv(os.path.join(data_path, "train_processed.csv"))
test_processed_data.to_csv(os.path.join(data_path, "test_processed.csv"))
