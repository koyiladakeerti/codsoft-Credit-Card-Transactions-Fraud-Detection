# codsoft-Credit-Card-Transactions-Fraud-Detection
Credit Card Transactions Fraud Detection
This project aims to detect fraudulent credit card transactions using machine learning. A Random Forest Classifier is trained on transactional data and evaluated for its ability to distinguish between fraudulent and non-fraudulent activity.
 Dataset
The dataset used in this project consists of two CSV files:

fraudTrain.csv
fraudTest.csv

Each row in the dataset represents a credit card transaction, with various attributes such as category, amount, location, demographic information, and a binary target variable is_fraud.
Data Preprocessing

Combined training and test datasets for consistent label encoding.
Dropped unnecessary or personally identifiable information.
Label encoded categorical variables: category, gender, and job.
Selected relevant features for training:

category
amt
gender
lat
long
city_pop
job

 Model Used

Random Forest Classifier

n_estimators = 100
random_state = 42



 Evaluation
After training the model, the following metrics were used to evaluate its performance on the test set:

Accuracy Score
Classification Report (Precision, Recall, F1-Score)
Confusion Matrix

Results
The model provides a strong baseline for fraud detection using basic engineered features and default hyperparameters of a Random Forest Classifier. Performance may be further improved through:

Hyperparameter tuning
Handling class imbalance
Advanced feature engineering

Requirements

Python 3.x
pandas
numpy
matplotlib
seaborn
scikit-learn

Install dependencies:
pip install pandas numpy matplotlib seaborn scikit-learn
