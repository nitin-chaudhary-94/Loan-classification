# Loan-classification
Introduction
In this case study, we are attempting to solve a real-world business problem using Exploratory Data Science techniques. 
We will be understanding and solving a risk analytics problem in Banking and Financial Domain.
We will be checking how data can be used effectively to solve business problems like defaulter’s prediction in Loan Lending club.

Abstract
Loan Classification using K-Nearest Neighbors (KNN) and Data Preprocessing

The financial sector's growth has fueled the need for accurate credit risk assessment methods, prompting the utilization of machine learning techniques in loan classification. This project focuses on developing a loan classification model using K-Nearest Neighbors (KNN) algorithm and encompasses data preprocessing, feature engineering, and model training.

The initial phase involves loading and preprocessing a loan dataset, including selecting relevant columns, converting categorical variables to numerical representations using label encoding, and handling missing values by imputing with medians. Selected numerical features are then scaled using StandardScaler to ensure consistent ranges.

A KNN classification model is trained using the scaled numerical features and their corresponding loan statuses. The model's performance is evaluated on a test set, calculating accuracy as a metric of classification success. The KNN model becomes capable of predicting the loan status of new data points by leveraging the trained patterns in the dataset.

This project showcases a comprehensive pipeline for loan classification, including data preprocessing, feature transformation, and KNN model training. The developed model offers insights into the applicability of KNN in credit risk assessment, contributing to the ongoing efforts to enhance the efficiency and accuracy of loan classification methods.

Keywords: Loan classification, K-Nearest Neighbors, credit risk assessment, data preprocessing, feature scaling, machine learning.

Business Problem Statement:
We are working for a consumer finance company which specialises in lending various types of loans to urban customers. 
When the company receives a loan application, the company has to make a decision for loan approval based on the applicant’s profile. 
Two types of risks are associated with the bank’s decision:
•	If the applicant is likely to repay the loan, then not approving the loan results in a loss of business to the company
•	If the applicant is not likely to repay the loan, i.e. he/she is likely to default, then approving the loan may lead to a financial loss for the company

The data given to us contains the information about past loan applicants and whether they ‘defaulted’ or not. 
The aim is to identify patterns which indicate if a person is likely to default, which may be used for taking actions such as denying the loan, reducing the amount of loan, lending (to risky applicants) at a higher interest rate, etc.

Business Objectives:
When a person applies for a loan, there are two types of decisions that could be taken by the company:
Loan accepted: If the company approves the loan, there are 3 possible scenarios described below:
•	Fully paid: Applicant has fully paid the loan (the principal and the interest rate)
•	Current: Applicant is in the process of paying the instalments, i.e. the tenure of the loan is not yet completed. These candidates are not labelled as 'defaulted'.
•	Charged-off: Applicant has not paid the instalments in due time for a long period of time, i.e. he/she has defaulted on the loan

Loan rejected: The company had rejected the loan (because the candidate does not meet their requirements etc.). 
Since the loan was rejected, there is no transactional history of those applicants with the company and so this data is not available with the company (and thus in this dataset)



SNAPSHOT OF DATASET

 
 
DATA DICTIONARY
LoanStatNew		Data type	Description
addr_state		object	The state provided by the borrower in the loan application
annual_inc		float64	The self-reported annual income provided by the borrower during registration.
collection_recovery_fee		float64	post charge off collection fee
delinq_2yrs		int64	The number of 30+ days past-due incidences of delinquency in the borrower's credit file for the past 2 years
dti		float64	A ratio calculated using the borrower’s total monthly debt payments on the total debt obligations, excluding mortgage and the requested LC loan, divided by the borrower’s self-reported monthly income.
earliest_cr_line		object	The month the borrower's earliest reported credit line was opened
emp_length		int64	Employment length in years. Possible values are between 0 and 10 where 0 means less than one year and 10 means ten or more years.
emp_title		object	The job title supplied by the Borrower when applying for the loan.*
funded_amnt		int64	The total amount committed to that loan at that point in time.
funded_amnt_inv		float64	The total amount committed by investors for that loan at that point in time.
grade		object	LC assigned loan grade
home_ownership		object	The home ownership status provided by the borrower during registration. Our values are: RENT, OWN, MORTGAGE, OTHER.
inq_last_6mths		int64	The number of inquiries in past 6 months (excluding auto and mortgage inquiries)
installment		float64	The monthly payment owed by the borrower if the loan originates.
int_rate		float64	Interest Rate on the loan
issue_d		object	The month which the loan was funded
last_credit_pull_d		object	The most recent month LC pulled credit for this loan
last_pymnt_amnt		float64	Last total payment amount received
last_pymnt_d		object	Last month payment was received
loan_amnt		int64	The listed amount of the loan applied for by the borrower. If at some point in time, the credit department reduces the loan amount, then it will be reflected in this value.
loan_status		object	Current status of the loan
open_acc		int64	The number of open credit lines in the borrower's credit file.
out_prncp		float64	Remaining outstanding principal for total amount funded
out_prncp_inv		float64	Remaining outstanding principal for portion of total amount funded by investors
pub_rec		int64	Number of derogatory public records
pub_rec_bankruptcies		object	Number of public record bankruptcies
purpose		object	A category provided by the borrower for the loan request.
recoveries		float64	post charge off gross recovery
revol_bal		int64	Total credit revolving balance
revol_util		float64	Revolving line utilization rate, or the amount of credit the borrower is using relative to all available revolving credit.
sub_grade		object	LC assigned loan subgrade
term		object	The number of payments on the loan. Values are in months and can be either 36 or 60.
title		object	The loan title provided by the borrower
total_acc		int64	The total number of credit lines currently in the borrower's credit file
total_pymnt		float64	Payments received to date for total amount funded
total_pymnt_inv		float64	Payments received to date for portion of total amount funded by investors
total_rec_int		float64	Interest received to date
total_rec_late_fee		float64	Late fees received to date
total_rec_prncp		float64	Principal received to date
verification_status		object	Indicates if income was verified by LC, not verified, or if the income source was verified
zip_code		object	The first 3 numbers of the zip code provided by the borrower in the loan application.


CONTENTS OF CODE

1.	Data Loading and Preprocessing:
   - The code loads a dataset from a CSV file named "loan_cleaned.csv."
   - It selects specific columns from the dataset for analysis.
   - Some columns like 'issue_d' and 'last_pymnt_d' are converted to datetime format, and the 'term' column is processed to extract the numeric part.

2. Label Encoding:
   - The code applies label encoding to selected categorical columns, converting them into numerical values. The columns include 'grade', 'sub_grade', 'emp_title', 'home_ownership', 'verification_status', 'purpose', and 'loan_status'.

3. Handling Missing Values:
   - The code fills missing values with the median value for all columns.

4. Feature Scaling:
   - The code uses StandardScaler to scale the selected numerical features.

5. K-Nearest Neighbors (KNN) Model:
   - The code initializes a KNN classifier with a specified number of neighbors.
   - The scaled training data is used to train the KNN model.
   - The model predicts the labels of the scaled test data.
   - The accuracy of the model is calculated using accuracy score.

6. Predicting New Data Point:
   - A new data point is created as a DataFrame with specific attribute values.
   - The new data point is preprocessed and scaled using the same scaler as the training data.
   - The trained KNN model is used to predict the loan status for the new data point.

 
FUNCTIONS USED IN THE DATASET


df.info():
•	df.info() is a function in pandas that provides a concise summary of a DataFrame's structure. It includes information about the number of non-null values in each column, the data type of each column, the memory usage, and more.
•	This function is useful for quickly assessing the data quality, identifying missing values, and understanding the overall composition of the DataFrame.
•	It helps you gain insights into the data types and memory usage, which can be important for optimizing memory consumption and data manipulation.


df.duplicated():
•	df.duplicated() is a pandas function that identifies duplicated rows in a DataFrame. It returns a Boolean Series that indicates whether each row is a duplicate of a previous row.
•	By default, it considers all columns for identifying duplicates. You can specify specific columns using the subset parameter if you want to check for duplicates based on a subset of columns.
•	Duplicated rows are marked as True in the Boolean Series, and unique (non-duplicated) rows are marked as False.


df.nunique():
The df.nunique() function is used to count the number of unique values in each column of a pandas DataFrame. It returns a Series with the count of distinct values for each column. This function is particularly useful for understanding the diversity of values present in your dataset and can provide insights into the variability and distribution of data.

df.shape:
The df.shape attribute in pandas returns a tuple representing the dimensions of a DataFrame. The tuple contains two values: the number of rows and the number of columns in the DataFrame. It provides a quick way to get an overview of the size of your DataFrame.


The expression (df.loan_status.value_counts()*100)/len(df) calculates the percentage distribution of each unique value in the "loan_status" column of a pandas DataFrame. It shows the proportion of each value relative to the total number of rows in the DataFrame.

The seaborn and matplotlib.pyplot libraries, which are commonly used for data visualization in Python. seaborn is built on top of matplotlib and provides a higher-level interface for creating informative and attractive statistical graphics.
ggplot: A Python implementation of the popular ggplot2 library from R.

sklearn.model_selection :

sklearn.model_selection is a module within the scikit-learn library (sklearn) that provides a variety of tools for splitting datasets into train and test sets, selecting and tuning model hyperparameters, and evaluating model performance. This module is essential for designing effective machine learning workflows, ensuring that your models generalize well to new, unseen data.

Train-Test Splitting:
•	This function splits a dataset into training and testing subsets, which are used to train and evaluate machine learning models, respectively.

pd.get_dummies() is a function provided by the pandas library that is used for creating binary (0 or 1) indicator variables from categorical variables. It's a common preprocessing step in data analysis and machine learning to convert categorical variables into a format that can be used by various machine learning algorithm

K-Nearest Neighbors (KNN):
K-Nearest Neighbors (KNN) is a simple and widely used classification and regression algorithm in machine learning. It's a type of instance-based learning where new data points are classified based on the majority class of their k-nearest neighbors in the feature space. In other words, it assumes that data points with similar features tend to belong to the same class.






 
DATA CLEANING

Cleaning a Dataset is a most  important aspect of Analyzation.

Steps Involve in Cleaning are :

1.	Handling Missing Values:
•	Identify and understand the presence of missing values in your dataset.
•	Decide how to handle missing values: impute them (fill with estimated values) or remove rows/columns with missing values.
•	Common imputation methods include mean, median, mode imputation or more advanced methods like regression-based imputation.
2.	Handling Duplicates:
•	Detect and remove duplicate records from your dataset to ensure each observation is unique.
•	Duplicates can skew analyses and model results if not handled properly.
3.	Dealing with Outliers:
•	Identify outliers, which are data points that significantly deviate from the rest of the data.
•	Decide whether to keep, transform, or remove outliers based on domain knowledge and the impact on your analysis.
4.	Data Type Conversion:
•	Ensure that each column has the appropriate data type (e.g., numeric, categorical, datetime) for analysis and modeling.
•	Convert data types as needed to ensure consistency and proper treatment.
5.	Handling Inconsistent Data and Typos:
•	Detect and correct inconsistent or erroneous data, such as typos or inconsistent capitalization.
•	Use tools like regular expressions or fuzzy matching to standardize data.
6.	Standardizing and Normalizing:
•	Standardize categorical data by ensuring consistent categories and labels.
•	Normalize numerical data to bring values to a common scale, which can be important for some algorithms.
7.	Handling Text and Categorical Data:
•	Convert text data into a format suitable for analysis (e.g., tokenization, stemming, lemmatization).
•	Encode categorical variables using techniques like one-hot encoding or label encoding.
8.	Data Integrity and Consistency:
•	Validate data integrity by checking for logical inconsistencies or violations of business rules.
•	Ensure data consistency across different sources or datasets.
9.	Data Integration:
•	Merge or join multiple datasets if needed, ensuring proper alignment of data.
10.	Data Reduction:
•	Reduce dimensionality if necessary using techniques like Principal Component Analysis (PCA).
11.	Visual Inspection:
•	Visualize cleaned data using plots and graphs to identify any remaining issues or anomaly

 
DATA PREPROCESSING AND ANALYSIS

Data preprocessing is a crucial step in preparing a dataset for analysis. It involves various operations to ensure that the data is accurate, consistent, and suitable for further analysis and modeling. Here's a breakdown of the pre-processing steps involved in the provided code snippet:

1. Handling Missing Values:
•	Identify missing values in the dataset.
•	Choose appropriate strategies to handle missing values, such as imputation with median or mean.
•	Ensure missing values are properly filled to prevent skewed results.

2. Handling Categorical Data:
•	Convert 'term' column into an integer representing loan term in months.
•	Apply label encoding to categorical columns like 'grade', 'sub_grade', 'emp_title', 'home_ownership', 'verification_status', 'purpose', 'loan_status'.

3. Standardization:
•	Standardize numerical features using 'StandardScaler'.
•	Selected numerical columns are scaled to have consistent scales for accurate modeling.

4. Model Training and Evaluation:
•	Initialize a K-Nearest Neighbors (KNN) classification model with 'n_neighbors=3'.
•	Split the data into training and test sets using 'train_test_split'.
•	Fit the KNN model to the scaled training data.
•	Predict loan statuses for the test data using the trained model.
•	Calculate and print the accuracy score of the model's predictions.

5. Predicting with New Data:
•	Create a sample new data point with values for selected numerical features.
•	Preprocess and transform the new data point using the same scaler.
•	Use the trained KNN model to predict the loan status for the new data point.
•	Print the predictions for the new data point.

In summary, the provided code snippet focuses on pre-processing loan data by handling missing values, converting categorical data, standardizing numerical features, training a KNN classification model, and predicting loan statuses for both test data and new data points. The goal is to prepare the data for accurate analysis and prediction of loan outcomes.

 
Checking Outliers: Annual Income

Removing outliers from data is important to improve model accuracy, mitigate bias, enhance interpretability, reduce noise, ensure robustness, preserve assumptions, prevent overfitting, and enhance data quality.

 


After Removing Outliers





 
VISUALIZATIONS


 

The Subplot made in Python for following features.
•	loan_amnt 
•	funded_amnt
•	funded_amnt_inv



Dashboard of Loan Classification in Tableau

 


For Better Understanding of data, here is a Dashboard Created in Tableau. Representing Different aspects of data Visually.
 
Purpose and Recoveries

 

Bubble Chart representing the purpose of the loan i.e., for what reason the customer has taken the loan. Here the size of the bubble represents the Number of Loans for the Specific Purpose with the Total Recoveries 


Loan Amount and Installment

 

Line graph Symbolizes Loan Amount and installments over the period. Here we can see a semi parabola, Over the period of time  an increase in amount can be observed for both. 

 
Recoveries for Home Loans

 

The Horizontal graph is visual representation of Recoveries of Loans related to Home. It covers Mortgage, Rent, Ownership and other attributes. The color shows the number of loans more the loans darker the color will be. Mortgage is highest in number.

Loan Status

 

Loan status and total Rec Late Fee is Graph between total late fees received till date and status of the loan. It has 3 types of status, Fully Paid Charged off and current. 
•	Fully paid: Applicant has fully paid the loan (the principal and the interest rate)
•	Current: Applicant is in the process of paying the instalments, i.e. the tenure of the loan is not yet completed. These candidates are not labelled as 'defaulted'.
•	Charged-off: Applicant has not paid the instalments in due time for a long period of time, i.e. he/she has defaulted on the loan

 
MAP

 

Map is representing the States of America with most funded amount, Here California is most funded state, whereas, Northern states have less Funding.

Loan Amount Over the Years

 


The Line Graph Clearly Shows that the Loan Amount is increasing over the period, between the Year 2007 and 2011 it has increased from few thousands to nearly 200 Million.


 
CONCLUSION

1.	Data Cleaning and Pre-processing:
•	The initial steps involve loading the dataset and performing data cleaning to ensure the dataset's integrity.
•	Columns with either no unique values or a single unique value are dropped as they may not contribute useful information for analysis.
•	Missing values are addressed in columns like 'emp_length' and 'pub_rec_bankruptcies' by filling them with appropriate values.
•	Symbols like '%' are removed from columns like 'int_rate' and 'revol_util' to convert them into numeric data for analysis.
•	Categorical attributes are appropriately converted into categorical data types to improve data representation.


2.	Data Visualization:
•	Visualizations, including boxplots and distribution plots, provide insights into the distributions of important attributes like 'annual_inc', 'loan_amnt', 'funded_amnt', and more.
•	Boxplots reveal the presence of outliers and the overall spread of data, aiding in understanding the central tendencies and variability of key features.

3.	Machine Learning Model (K-Nearest Neighbors):
•	The code transitions into machine learning by utilizing the K-Nearest Neighbors (KNN) algorithm for classification.
•	The target variable 'loan_status' is set, and input features are preprocessed and encoded.
•	The dataset is split into training and testing sets to evaluate the model's performance effectively.
•	Standardization is applied to the feature data to ensure fair comparisons during distance-based calculations.
•	The KNN model is trained on the training data, and predictions are made on the test data.
•	Model accuracy is calculated, providing an initial assessment of the model's effectiveness.
•	A classification report is generated, offering comprehensive metrics on precision, recall, F1-score, and support for each class.

In conclusion, this code represents a well-structured and systematic approach to data analysis and machine learning. It encompasses data cleaning, exploratory data visualization, and the implementation of a K-Nearest Neighbors classification model. The insights gained from data visualization aid in understanding the dataset's characteristics, while the KNN model allows for predictive analysis based on the dataset's features. Further refinement and tuning of the machine learning model would be beneficial for optimizing predictive accuracy and achieving robust results in real-world scenarios.
 
References:

•	https://colab.research.google.com/
•	https://www.kaggle.com/datasets/abhishek14398/loan-dataset?select=loan.csv


