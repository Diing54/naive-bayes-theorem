#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#importing necessary libraries
import pandas as pd
import re
import string
from collections import defaultdict
import math
from sklearn.model_selection import train_test_split


# In[3]:


#Loading the dataset

url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
df = pd.read_csv(url, sep="\t", header=None, names=["label", "message"])
df["label"] = df["label"].map({"ham": 0, "spam": 1})  # Convert to binary


# In[4]:


#preprocessing

def preprocess(text):
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)  # remove punctuation
    return text.split()

df["tokens"] = df["message"].apply(preprocess)


# In[5]:


# Splitting the data

train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)


# In[6]:


#  Naive bayes classifier

class NaiveBayesClassifier:
    def __init__(self):
        self.word_freqs = {0: defaultdict(int), 1: defaultdict(int)}
        self.class_counts = defaultdict(int)
        self.vocab = set()

    def train(self, data):
        for i, row in data.iterrows():
            label = row["label"]
            tokens = row["tokens"]
            self.class_counts[label] += 1
            for word in tokens:
                self.word_freqs[label][word] += 1
                self.vocab.add(word)

    def predict(self, tokens):
        total_docs = sum(self.class_counts.values())
        log_probs = {}

        for c in [0, 1]:  # 0 = ham, 1 = spam
            log_prob = math.log(self.class_counts[c] / total_docs)
            total_words = sum(self.word_freqs[c].values())

            for word in tokens:
                word_count = self.word_freqs[c][word]
                prob = (word_count + 1) / (total_words + len(self.vocab))  # Laplace smoothing
                log_prob += math.log(prob)

            log_probs[c] = log_prob

        return 1 if log_probs[1] > log_probs[0] else 0

    def evaluate(self, test_data):
        correct = 0
        total = len(test_data)
        for _, row in test_data.iterrows():
            prediction = self.predict(row["tokens"])
            if prediction == row["label"]:
                correct += 1
        return correct / total


# In[7]:


# Training 

nb = NaiveBayesClassifier()
nb.train(train_data)

accuracy = nb.evaluate(test_data)
print(f"\n\u2705 Accuracy: {accuracy:.4f}")


# In[12]:


#Sample prediction 


def predict_sample_message(message):
    cleaned = preprocess(message)
    prediction = nb.predict(cleaned)
    return "spam" if prediction == 1 else "ham"

# Example sample message
sample_message = "Click the link below to win a brand new iphone"
prediction = predict_sample_message(sample_message)

print(f"\nSample Message: '{sample_message}'")
print(f"Prediction: {prediction}")

from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

# Get predictions for all test data
y_true = test_data["label"].tolist()
y_pred = [nb.predict(row["tokens"]) for _, row in test_data.iterrows()]

# Calculate metrics
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")

import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Ham", "Spam"], yticklabels=["Ham", "Spam"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

