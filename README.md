# Phishing-Website-Detection-Using-Machine-Learning

This project addresses the rising threat of phishing websites used by cybercriminals to deceive users and steal sensitive information. By leveraging machine learning models, our system aims to accurately classify websites as either legitimate or phishing.

# **Problem Statement**

Phishing websites mimic trusted URLs and webpages to trick users into revealing personal data. With the growth of online services, automated and efficient methods for detecting these websites are essential. This project provides a machine learning-based approach to identify and classify phishing websites in real time.


# **Objectives**

Develop machine learning models to accurately predict phishing websites using features extracted from URLs.

Compare the performance of various algorithms and identify the best model for this classification task.

Create a scalable and automated system that reduces reliance on manual or rule-based detection methods.


# **Dataset Overview**

We used a dataset containing features extracted from 10,000 URLs, both legitimate and phishing. The features include:

Domain: The domain name of the website.

Have_IP: Indicates the use of an IP address in the URL (binary).

URL_Length: Length of the URL.

https_Domain: Use of HTTPS in the domain (binary).

... (and others, listing relevant features).

The dataset is balanced and preprocessed to ensure fair training and testing.


#**Methodology**

**Programming Language**: Python, using Google Colab for development and result visualization.

**Libraries:** Pandas, NumPy, Scikit-learn, and XGBoost for data processing and machine learning.

**Data Preprocessing:** Performed data cleaning, feature extraction, and train-test split (80:20 ratio).


# **Machine Learning Models**

**Decision Tree:** A simple model for classification based on a series of decision rules.

**Random Forest:** An ensemble of decision trees, reducing overfitting and improving accuracy.

**Multilayer Perceptrons (MLP):** Neural networks for complex pattern recognition.

**XGBoost:** A powerful gradient boosting model that provided the best performance in our tests.

**Support Vector Machines (SVM):** A binary classifier that separates data using a hyperplane.

**Autoencoder:** A neural network used for feature reduction and anomaly detection.


# **Model Evaluation**

The models were compared based on accuracy, precision, recall, and F1-score.

Best Performing Model: The XGBoost Classifier achieved the highest accuracy of 86% on the testing dataset.


# **Results and Insights**

XGBoost: Most effective model with the highest accuracy and minimal overfitting.

Random Forest and SVM: Also performed well but required more fine-tuning.

Autoencoder: Useful for feature extraction but less effective as a standalone classifier.


# **Visualizations**

Data distribution, feature correlations, and model performance metrics were visualized using various plots and graphs to understand the dataset and model behavior.


# **Future Work**

Feature Engineering: Explore additional features that could improve model accuracy.

Real-Time Implementation: Deploy the model for real-time phishing detection.

Advanced Models: Experiment with deep learning models for further accuracy improvements.


# Conclusion

This project successfully demonstrates the potential of machine learning for phishing website detection. By using XGBoost, we achieved an accuracy of 86%, highlighting the efficacy of automated solutions in cybersecurity. Future enhancements could make this approach even more robust and scalable.
