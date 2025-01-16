# Email-Spam-Detection-FederatedLearning
Developed an email spam detection system by implementing privacy preserving techniques such as Differencial privacy and Federated Learning to train models across decentralized data sources without sharing raw data. Also Implemented NLP’s nltk libraries for feature extraction.Enhanced data privacy and scalability in detecting spam .

			                                                               EMAIL SPAM DETECTION
                                                                                 (PRIVACY USING FEDERATED LEARNING)

1.Abstract
=
This project applies federated learning to build a robust spam detection system where model training is distributed across multiple clients, and only model parameters, not the raw data, are aggregated centrally. By machine learning classifiers, each client locally trains a spam detection model on their own email dataset. The central server then aggregates these updates to improve the global model without compromising user privacy.

2.Federated Learning
=
Federated Learning is a technique of training machine learning models on decentralized data, where the data is distributed across multiple devices or nodes, such as smartphones, IoT devices, edge devices, etc. Instead of centralizing the data and training the model in a single location, in Federated Learning, the model is trained locally on each device and the updates are then aggregated and shared with a central server.



3.Differential Privacy
=
Differential privacy is a privacy-preserving technique that conceals individual data points in a dataset by adding controlled random noise
while balancing privacy protection and data utility.

4.Email Spam Detection
=
Nowadays, a big part of people rely on available email or messages sent by the stranger. The possibility that anybody can leave an email or a message provides a golden opportunity for spammers to write spam message about our different interests. Spam emails are the emails that the receiver does not wish to receive. A large number of identical messages are sent to several recipients of email. Spam emails fill our Inbox with number of ridiculous emails , degrades our Internet speed to a great extent and steals useful information like our details on our Contact list. In our project our objective is to develop a privacy-preserving email spam detection system using federated learning, where individual user data remains on local devices and only model updates are shared.

5.Models
=
MLP - Use multiple layers of interconnected neurons to learn complex patterns in data for email classification.
SVM - Learn decision boundaries between spam and non-spam emails.
Logistic Regression - Model the probability that an email is spam or not.
Naive Bayes - Apply probabilistic reasoning using feature likelihoods for classification.
KNN -Classify emails based on the majority label among the nearest neighbours in the feature space.

6.Implementation
=
Implemented Federated Learning for all the above models and found MLP has higher accuracy, so implemented differential privacy for MLP and also implemented Federated learning with differential privacy integrated together.

7.Dataset
=
SpamAssassin Public Corpus
The SpamAssassin Public Corpus is a collection of publicly available emails used for spam detection research. It consists of 6,053 ham emails (legitimate) and 1,893 spam emails (unsolicited), with a total of 7,946 emails. This dataset is commonly used for training and testing spam filtering algorithms.

https://www.kaggle.com/datasets/beatoa/spamassassin-public-corpus


8.References
=

•	https://spamassassin.apache.org/doc.html

•	https://towardsdatascience.com/federated-learning-a-step-by-step-implementation-in-tensorflow-aac568283399


