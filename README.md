# naive-bayes-theorem
This repository contains a simple implementation of naive-bayes algorithm from scratch to understand the basic concepts. It aims to classify sms messages as either spam or not spam.
This is a probabilistic algorithm that assumes all features are independent of each other hence the name 'naive'.

1. Methodology
Dataset

    I used the SMS Spam Collection Dataset, sourced from a public GitHub repository.

    The dataset contains thousands of text messages labeled as either:

    "ham" (not spam), or
    "spam".

**Preprocessing**

  All text messages were lowercased to maintain uniformity.

  Punctuation was removed to avoid unnecessary noise.

  Each message was tokenized (split into words).

  No missing values were present in the dataset, so no imputation was needed.

  The label (ham/spam) was mapped to binary values (0 for ham, 1 for spam).

**Data Splitting**

The data was split into a training set (80%) and a testing set (20%) using train_test_split from sklearn.model_selection.

**Model Implementation**

A Naive Bayes Classifier was implemented from scratch using basic Python libraries.

Key components of the model:

Prior probabilities were calculated based on class distribution.

Likelihoods for each word given a class were calculated with Laplace smoothing to handle unseen words.

During prediction, log probabilities were used for numerical stability.

For a given message, the class with the higher total log probability was selected.

**Evaluation**

The model was evaluated on the test data using:

Accuracy, Precision, Recall and F1-score

A confusion matrix was plotted using seaborn and matplotlib to visually inspect classification performance.

2. Results

Model Performance
Metric	Score
Accuracy	~0.9803
Precision	~0.9150
Recall	~0.9396
F1-score	~0.9272

**Confusion Matrix**

Most messages were classified correctly.

![image](https://github.com/user-attachments/assets/a9a3adb9-2712-4761-8ed0-40eae2fdb990)


There were a few false positives (ham messages incorrectly classified as spam) and a few false negatives (spam missed as ham), but overall the model was very reliable.
