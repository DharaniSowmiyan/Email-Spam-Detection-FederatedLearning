# Email-Spam-Detection-FederatedLearning

Developed an email spam detection system by implementing privacy-preserving techniques such as Differential Privacy and Federated Learning to train models across decentralized data sources without sharing raw data. Also implemented NLP’s NLTK libraries for feature extraction. Enhanced data privacy and scalability in detecting spam.

## EMAIL SPAM DETECTION

### (PRIVACY USING FEDERATED LEARNING)

Federated Learning for Email Spam Classification

This repository implements federated learning for email spam classification using multiple machine learning models, including:

✅ Support Vector Machine (SVM)✅ Logistic Regression✅ Naïve Bayes✅ Random Forest

Each model is trained across multiple clients in a federated setting, where local models are trained separately and their predictions are aggregated globally.

📊 Approach

1️⃣ Preprocessing:

Uses TfidfVectorizer to convert email text into numerical feature vectors.

Splits data into training and testing sets.

2️⃣ Federated Learning Simulation:

The training data is split among num_clients (e.g., 5 clients).

Each client trains an independent model on its local data.

Predictions from all clients are aggregated to compute a global result.

3️⃣ Evaluation Metrics:

Accuracy

Precision

Recall

F1-Score

AUC-ROC Score

## 1. Abstract

This project applies federated learning to build a robust spam detection system where model training is distributed across multiple clients, and only model parameters, not the raw data, are aggregated centrally. Using machine learning classifiers, each client locally trains a spam detection model on their own email dataset. The central server then aggregates these updates to improve the global model without compromising user privacy.

## 2. Federated Learning

Federated Learning is a technique of training machine learning models on decentralized data, where the data is distributed across multiple devices or nodes, such as smartphones, IoT devices, edge devices, etc. Instead of centralizing the data and training the model in a single location, in Federated Learning, the model is trained locally on each device and the updates are then aggregated and shared with a central server.

## 3. Differential Privacy

Differential privacy is a privacy-preserving technique that conceals individual data points in a dataset by adding controlled random noise while balancing privacy protection and data utility.

## 4. Email Spam Detection

Nowadays, a big part of people rely on available email or messages sent by strangers. The possibility that anybody can leave an email or a message provides a golden opportunity for spammers to write spam messages about our different interests. Spam emails are the emails that the receiver does not wish to receive. A large number of identical messages are sent to several recipients of email. Spam emails fill our inbox with a number of ridiculous emails, degrade our Internet speed to a great extent, and steal useful information like our details on our contact list. In our project, our objective is to develop a privacy-preserving email spam detection system using federated learning, where individual user data remains on local devices and only model updates are shared.

## 5. Models

- **MLP** - Uses multiple layers of interconnected neurons to learn complex patterns in data for email classification.
- **SVM** - Learns decision boundaries between spam and non-spam emails.
- **Logistic Regression** - Models the probability that an email is spam or not.
- **Naive Bayes** - Applies probabilistic reasoning using feature likelihoods for classification.
- **KNN** - Classifies emails based on the majority label among the nearest neighbors in the feature space.

## 6. Implementation

Implemented Federated Learning for all the above models and found MLP has higher accuracy, so implemented differential privacy for MLP and also implemented Federated Learning with differential privacy integrated together.

## 7. Dataset

### SpamAssassin Public Corpus

The SpamAssassin Public Corpus is a collection of publicly available emails used for spam detection research. It consists of 6,053 ham emails (legitimate) and 1,893 spam emails (unsolicited), with a total of 7,946 emails. This dataset is commonly used for training and testing spam filtering algorithms.

[SpamAssassin Public Corpus Dataset](https://www.kaggle.com/datasets/beatoa/spamassassin-public-corpus)

## 8. References

- [SpamAssassin Documentation](https://spamassassin.apache.org/doc.html)
- [Federated Learning - A Step-by-Step Implementation](https://towardsdatascience.com/federated-learning-a-step-by-step-implementation-in-tensorflow-aac568283399)

