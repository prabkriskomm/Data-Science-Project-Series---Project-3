#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

# Specify encoding
train_df = pd.read_csv('train.csv', encoding='ISO-8859-1')
test_df = pd.read_csv('test.csv', encoding='ISO-8859-1')
manual_test_df = pd.read_csv('testdata.manual.2009.06.14.csv', encoding='ISO-8859-1')
large_train_df = pd.read_csv('training.1600000.processed.noemoticon.csv', encoding='ISO-8859-1')

# Explore the datasets
print("Train Data:")
print(train_df.head())
print(train_df.info())

print("\nTest Data:")
print(test_df.head())
print(test_df.info())

print("\nManual Test Data:")
print(manual_test_df.head())
print(manual_test_df.info())

print("\nLarge Train Data:")
print(large_train_df.head())
print(large_train_df.info())

# Check sentiment distribution in train and test datasets
print("\nTrain Data Sentiment Distribution:")
print(train_df['sentiment'].value_counts())

print("\nTest Data Sentiment Distribution:")
print(test_df['sentiment'].value_counts())


# In[5]:


import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download NLTK data
#nltk.download('stopwords')
#nltk.download('punkt')
#nltk.download('wordnet')

# Initialize lemmatizer and stop words
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Function for preprocessing text
def preprocess_text(text):
    text = text.lower()  # Lowercasing
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters
    words = word_tokenize(text)  # Tokenization
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]  # Lemmatization and stop word removal
    return ' '.join(words)

# Load datasets
train_df = pd.read_csv('train.csv', encoding='ISO-8859-1')
test_df = pd.read_csv('test.csv', encoding='ISO-8859-1')

# Fill missing values in 'text' column with an empty string
train_df['text'].fillna('', inplace=True)
test_df['text'].fillna('', inplace=True)

# Apply preprocessing to train and test data
train_df['cleaned_text'] = train_df['text'].apply(preprocess_text)
test_df['cleaned_text'] = test_df['text'].apply(preprocess_text)

# Check the results
print("Preprocessed Train Data:")
print(train_df[['text', 'cleaned_text']].head())
print("\nPreprocessed Test Data:")
print(test_df[['text', 'cleaned_text']].head())


# In[6]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Vectorize the text data using TF-IDF
tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(train_df['cleaned_text'])
X_test_tfidf = tfidf.transform(test_df['cleaned_text'])

# Extract labels
y_train = train_df['sentiment']
y_test = test_df['sentiment']

# Split the train data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train_tfidf, y_train, test_size=0.2, random_state=42)

# Print the shape of the datasets
print(f"Training Data Shape: {X_train.shape}")
print(f"Validation Data Shape: {X_val.shape}")
print(f"Test Data Shape: {X_test_tfidf.shape}")


# In[8]:


# Train a Naive Bayes classifier
nb = MultinomialNB()
nb.fit(X_train, y_train)

# Predict and evaluate on validation data
y_val_pred_nb = nb.predict(X_val)
print("Naive Bayes Validation Accuracy:", accuracy_score(y_val, y_val_pred_nb))
print(classification_report(y_val, y_val_pred_nb))

# Assuming `test_df` is your test DataFrame with the necessary columns
# Define `y_test` from the `test_df` if not already done
y_test = test_df['sentiment']

# Ensure `y_test` is a pandas Series of strings (sentiment labels)
y_test = y_test.astype(str)

# Check unique values in y_test to ensure correctness
print("Unique values in y_test:", y_test.unique())

# Now you can proceed with evaluating the model predictions
y_test_pred_nb = nb.predict(X_test_tfidf)

# Convert predictions to strings if necessary
y_test_pred_nb = y_test_pred_nb.astype(str)

# Evaluate the Naive Bayes model on test data
print("Naive Bayes Test Accuracy:", accuracy_score(y_test, y_test_pred_nb))
print(classification_report(y_test, y_test_pred_nb))


# In[10]:


# Train an SVM classifier
svm = SVC()
svm.fit(X_train, y_train)

# Predict and evaluate on validation data
y_val_pred_svm = svm.predict(X_val)
print("SVM Validation Accuracy:", accuracy_score(y_val, y_val_pred_svm))
print(classification_report(y_val, y_val_pred_svm))

# Predict and evaluate on test data
y_test_pred_svm = svm.predict(X_test_tfidf)
print("SVM Test Accuracy:", accuracy_score(y_test, y_test_pred_svm))
print(classification_report(y_test, y_test_pred_svm))


# In[11]:


# Train a Logistic Regression classifier
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)

# Predict and evaluate on validation data
y_val_pred_lr = lr.predict(X_val)
print("Logistic Regression Validation Accuracy:", accuracy_score(y_val, y_val_pred_lr))
print(classification_report(y_val, y_val_pred_lr))

# Predict and evaluate on test data
y_test_pred_lr = lr.predict(X_test_tfidf)
print("Logistic Regression Test Accuracy:", accuracy_score(y_test, y_test_pred_lr))
print(classification_report(y_test, y_test_pred_lr))


# In[12]:


from sklearn.model_selection import GridSearchCV

param_grid = {'alpha': [0.01, 0.1, 1]}
grid_search = GridSearchCV(MultinomialNB(), param_grid, cv=5)
grid_search.fit(X_train, y_train)
print("Best parameters for Naive Bayes:", grid_search.best_params_)


# In[13]:


from sklearn.model_selection import cross_val_score

nb_best_model = MultinomialNB(alpha=grid_search.best_params_['alpha'])
scores = cross_val_score(nb_best_model, X_train, y_train, cv=5)
print("Cross-Validation Scores:", scores)
print("Mean CV Score:", scores.mean())


# In[ ]:





# In[ ]:





# In[ ]:




