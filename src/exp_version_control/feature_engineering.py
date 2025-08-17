import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


train_data = pd.read_csv('data/processed/train_processed.csv')
test_data = pd.read_csv('data/processed/test_processed.csv')

train_data.fillna('', inplace=True)
test_data.fillna('', inplace=True)

x_train = train_data['content'].values
y_train = train_data['sentiment'].values

x_test = test_data['content'].values
y_test = test_data['sentiment'].values

# Apply Bag of words (CountVectorizer)
vectorizer = CountVectorizer(max_features=50)

# Fit the vectorizer on the training data
X_train_bow = vectorizer.fit_transform(x_train)

# Transform the test data
X_test_bow = vectorizer.transform(x_test)

train_df = pd.DataFrame(X_train_bow.toarray())
train_df['label'] = y_train
test_df = pd.DataFrame(X_test_bow.toarray())
test_df['label'] = y_test

# store the data inside data/features

data_dir = os.path.join("data", "features")
os.makedirs(data_dir, exist_ok=True)
train_df.to_csv(os.path.join(data_dir, "train_bow.csv"), index=False)
test_df.to_csv(os.path.join(data_dir, "test_bow.csv"), index=False)